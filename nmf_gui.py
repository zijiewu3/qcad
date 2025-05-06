# --- Standard library imports ---
import sys
import os
import json

# --- PyQt5 GUI components ---
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QListWidget, QHBoxLayout, QRadioButton, QLineEdit, QCheckBox, QComboBox,
    QButtonGroup, QTabWidget, QGridLayout
)
from PyQt5.QtCore import QTimer

# --- Scientific libraries ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pickle
import os
import glob
from skimage import exposure

def contrast_stretch(series,p1,p2):
  series_reshape = series.reshape(-1,series.shape[-2],series.shape[-1])
  transformed = np.array([exposure.rescale_intensity(im, (np.percentile(im,p1), np.percentile(im,p2))) for im in series])
  return transformed.reshape(series.shape)

def glob_ndatas_recursive(directory):
            """Globs all files under the given directory."""
            return glob.glob(directory + '/**/*ndata', recursive=True)

# --- Main GUI class ---
class NMFGui(QWidget):
    def __init__(self, data_folder=os.getcwd()):
        super().__init__()
        self.setWindowTitle("NMF Model Deployer")  # Set the window title
        self.tabs = QTabWidget()
        layout = QVBoxLayout()                     # Main vertical layout
        self.setLayout(layout)                    # Set the main layout for the widget
        layout.addWidget(self.tabs)  # Add the tab widget to your main layout

        self.data_tab = QWidget()
        self.data_tab_layout = QVBoxLayout()
        self.data_tab.setLayout(self.data_tab_layout)

        self.tabs.addTab(self.data_tab, "Data")  # Add a tab for data

        self.model_tab = QWidget()
        self.model_tab_layout = QVBoxLayout()   
        self.model_tab.setLayout(self.model_tab_layout)
        self.tabs.addTab(self.model_tab, "Model")


        self.data_folder = data_folder             # Folder to watch for .npy files
        self.selected_file = None                  # Store selected data file path
        self.selected_model = None                 # Store selected model object
        self.file_dict = {}                        # Dictionary to store file paths and metadata
        self.folder_label = QLabel(f"current folder: {self.data_folder}") # folder label
        # Label for clarity
        self.folder_type_label = QLabel("Folder Type:")
        self.radio_static = QRadioButton("Static")
        self.radio_watcher = QRadioButton("Watching")

        self.folder_type_group = QButtonGroup()
        self.folder_type_group.addButton(self.radio_static, id=0)
        self.folder_type_group.addButton(self.radio_watcher, id=1)


        # Connect to handler
        self.folder_type_group.buttonClicked[int].connect(self.set_folder_type)

        # Initialize the value
        self.folder_type = "static"

        # Set one as default (optional)
        self.radio_static.setChecked(True)

        # --- Layouts ---
        
        file_model_layout = QVBoxLayout()          # Layout for file list and model button

        # --- File watcher UI ---
        self.file_list_label = QLabel("Select a data file:")
        self.file_list = QListWidget()             # Widget to display list of files
        self.refresh_file_list()                   # Populate it with current file list
        self.file_list.itemClicked.connect(self.select_file)  # When a file is clicked, call select_file()
        # --- data folder button ---
        self.select_folder_button = QPushButton("Select Data Folder")
        self.select_folder_button.clicked.connect(self.choose_data_folder)
        
        # --- Model loader button ---
        self.model_button = QPushButton("Load NMF Model")
        self.model_button.clicked.connect(self.load_model)  # Connect button click to model loader

        # --- Model name label ---
        self.model_name_label = QLabel("No model loaded")  # Label to show loaded model name

        # --- selected file label ---
        self.selected_file_label = QLabel("No file selected")  # Label to show selected file name

        # --- Run layout ---
        self.run_layout = QHBoxLayout()

        # --- Autorun checkbox ---
        self.autorun_checkbox = QCheckBox("Autorun")
        self.autorun_checkbox.stateChanged.connect(self.toggle_autorun)

        # --- Timer for autorun ---
        self.autorun_timer = QTimer(self)
        self.autorun_timer.timeout.connect(self.run_model)
        # --- Run button ---
        self.run_button = QPushButton("Run Model")
        self.run_button.clicked.connect(self.run_model)     # Connect to model runner

        self.run_layout.addWidget(self.run_button)  # Add run button to layout
        self.run_layout.addWidget(self.autorun_checkbox)




        # --- Matplotlib plot embedded in PyQt ---
        self.canvas_layout = QGridLayout()
        self.figure1, self.ax1 = plt.subplots()
        self.canvas1 = FigureCanvas(self.figure1)

        self.figure2, self.ax2 = plt.subplots(constrained_layout=False)
        self.canvas2 = FigureCanvas(self.figure2)

        self.figure3, self.ax3 = plt.subplots()
        self.canvas3 = FigureCanvas(self.figure3)

        self.figure4, self.ax4 = plt.subplots()
        self.canvas4 = FigureCanvas(self.figure4)

        self.canvas_layout.addWidget(self.canvas1, 0, 0)
        self.canvas_layout.addWidget(self.canvas2, 0, 1)
        self.canvas_layout.addWidget(self.canvas3, 1, 0)
        self.canvas_layout.addWidget(self.canvas4, 1, 1)

        self.canvas_layout.setRowStretch(0, 3)
        self.canvas_layout.setRowStretch(1, 2)

        # --- Assemble layouts ---
        file_model_layout.addWidget(self.folder_type_label)
        file_model_layout.addWidget(self.radio_static)
        file_model_layout.addWidget(self.radio_watcher)

        file_model_layout.addWidget(self.file_list)
        file_model_layout.addWidget(self.model_button)
        file_model_layout.addWidget(self.model_name_label)
        

        self.data_tab_layout.addWidget(self.select_folder_button)
        self.data_tab_layout.addWidget(self.folder_label)
        self.data_tab_layout.addWidget(self.file_list_label)
        self.data_tab_layout.addLayout(file_model_layout)

        self.model_tab_layout.addWidget(self.selected_file_label)
        self.model_tab_layout.addWidget(self.model_name_label)
        self.model_tab_layout.addLayout(self.run_layout)
        self.model_tab_layout.addLayout(self.canvas_layout)
        

        #self.setLayout(layout)
        

        # --- Timer to auto-refresh file list every 5 seconds ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_file_list)
        self.timer.start(5000)

    # --- Refresh file list from folder ---
    def watch_all_ndatas(self):
        file_dict = {}
        self.file_list.clear()
        all_ndata_files = glob_ndatas_recursive(self.data_folder)
        for file in all_ndata_files:
            #data4D      = np.load(file)['data']
            data4D_meta = json.loads(np.load(file)['metadata.json'])
            file_dict[data4D_meta['title']] = file
        self.file_dict = file_dict

    def refresh_file_list(self):
        self.file_list.clear()
        if self.folder_type == "static":
            for fname in os.listdir(self.data_folder):
                if fname.endswith(".ndata1"):                 # Look for microscope data files
                    self.file_list.addItem(fname)
        if self.folder_type == "watching":
            self.watch_all_ndatas()
            for fname in self.file_dict.keys():
                self.file_list.addItem(fname)

    def set_folder_type(self, id):
        if id == 0:
            self.folder_type = "static"
            print("Folder type set to static.")
            self.refresh_file_list()  # Refresh file list when switching to static
        elif id == 1:
            self.folder_type = "watching"
            print("Folder type set to watching.")
            self.refresh_file_list()  # Refresh file list when switching to watching
        else:
            print("Invalid folder type selected.")

    # --- Save selected file path ---
    def select_file(self, item):
        if self.folder_type == "static":
            self.selected_file = os.path.join(self.data_folder, item.text())
            print(f"Selected file: {self.selected_file}")
        elif self.folder_type == "watching":
            self.selected_file = self.file_dict[item.text()]
            print(f"Selected file: {self.selected_file}")
        self.selected_file_label.setText(f"Selected file: {os.path.basename(self.selected_file)}")

    # --- select data folder ---
    def choose_data_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder", "")
        if folder:
            self.data_folder = folder
            self.refresh_file_list()  # Update file list with the new folder
            self.folder_label.setText(f"Current folder: {self.data_folder}")
            print(f"Data folder set to: {self.data_folder}")

    # --- Load NMF model from .pkl file ---
    def load_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Select NMF Model", "", "Pickle Files (*.pkl)")
        if model_path:
            with open(model_path, 'rb') as f:
                self.selected_model = pickle.load(f)
            print(f"Loaded model from: {model_path}")
            self.model_name_label.setText(f"Model loaded: {os.path.basename(model_path)}")
            H = self.selected_model.components_
            n_components = H.shape[0]
            H_images = H.reshape((n_components, int(np.sqrt(H.shape[-1])), int(np.sqrt(H.shape[-1]))))
            self.figure4.clear()
            colors = plt.cm.tab10.colors
            for imi, im in enumerate(H_images):
                ax4 = self.figure4.add_subplot(2, n_components//2+1, imi+1)
                ax4.imshow(im, cmap='gray')
                ax4.set_axis_off()
                ax4.set_title(f"Component {imi+1}", color = colors[imi])
            self.figure4.tight_layout()
            self.canvas4.draw()

    # --- Run selected model on selected data file ---
    def run_model(self):
        if self.selected_file and self.selected_model:
            
            data = np.load(self.selected_file)['data']                # Load .npy file
            if len(data.shape) != 5:
                print("Data shape is not 5D. Please select a valid file.")
                return 
            data_preprocessed = self.preprocess()         # Apply preprocessing (your code can go here)
            print('time series.shape', self.time_series.shape)
            self.figure2.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
            self.figure1.clear()
            self.figure2.clear()
            self.figure3.clear()
            self.ax3 = self.figure3.add_subplot(1,1,1)
            label_handles = []
            colors = plt.cm.tab10.colors
            mse_colors = plt.cm.Set2.colors
            for i in range(self.time_series.shape[1]):
                for j in range(self.time_series.shape[2]):
                    current_series = self.time_series[:,i,j,:,:]
                    nmf_embedded = self.selected_model.transform(current_series.reshape(current_series.shape[0], -1))
                    reconstructed = nmf_embedded @ self.selected_model.components_
                    mse = np.mean((current_series.reshape(current_series.shape[0], -1) - reconstructed)**2, axis=1)
                    nmf_embedded = nmf_embedded/np.linalg.norm(nmf_embedded, axis=1, keepdims=True)
                    #W = self.selected_model.transform(data_preprocessed)  # Apply model
                    #H = self.selected_model.components_

                    # Plot the result (W @ H is the reconstructed signal)
                    
                    ax1 = self.figure1.add_subplot(self.time_series.shape[1],self.time_series.shape[2],i*self.time_series.shape[2]+j+1)
                    ax1.imshow(current_series[-1], cmap='gray')
                    ax1.set_axis_off()
                    ax2 = self.figure2.add_subplot(self.time_series.shape[1],self.time_series.shape[2],i*self.time_series.shape[2]+j+1)
                    lines = []
                    for dim in range(nmf_embedded.shape[1]):
                        line, = ax2.plot(nmf_embedded[:,dim], label = f'dim {dim}', color=colors[dim])
                        lines.append(line)

                    if not label_handles:
                        label_handles = lines

                    self.ax3.plot(mse, label=f"mse {i},{j}", color=mse_colors[i*self.time_series.shape[2]+j])
                    
            self.figure2.legend(
                handles=label_handles,
                loc='center left',
                bbox_to_anchor=(0.9, 0.5),
                #title='Dimensions'
            )
            self.figure3.legend()
            

            self.ax1.set_title("final frame")
            self.ax2.set_title("NMF embedding")
            self.ax3.set_title("MSE")
                    #self.ax.imshow(W @ H, aspect='auto', cmap='viridis')
                    #self.ax.set_title("Reconstructed Signal")
            self.figure1.tight_layout()
            self.figure3.tight_layout()
            self.figure4.tight_layout()
            #self.figure2.tight_layout(rect=[0, 0, 0.85, 1])
            self.canvas1.draw()
            self.canvas2.draw()
            self.canvas3.draw()
            self.canvas4.draw()
        else:
            print("Please select both a data file and a model.")
    
    def toggle_autorun(self):
        # Set up a timer
        
        if self.autorun_checkbox.isChecked():
            # Start the timer to call `run_model` every 2 seconds
            self.autorun_timer.start(2000)  # 2000 ms = 2 seconds
        else:
            # Stop the timer
            self.autorun_timer.stop()

    
    def preprocess(self):
        import json
        import glob
        from skimage.transform import resize
        def resize_images_skimage(arr):
            # Resize each image to 128x128 while keeping other dimensions
            return resize(arr, (*arr.shape[:-2], 128, 128), anti_aliasing=True, preserve_range=True).astype(arr.dtype)

        def glob_files(directory):
            """Globs all files under the given directory."""
            return glob.glob(directory + '/*ndata1', recursive=True)
        
        filename_4D = self.selected_file
        data4D      = np.load(filename_4D)['data']
        data4D_meta = json.loads(np.load(filename_4D)['metadata.json'])

        data_max = data4D.max()
        data_min = data4D.min()
        #print(data_max, data_min)
        data4D = (data4D - data_min)/(data_max - data_min)
        #print(data4D.shape)
        resized = resize_images_skimage(data4D)
        resized = np.reshape(resized, (-1, data4D.shape[1], data4D.shape[2], 128, 128))
        
        self.time_series = contrast_stretch(resized, 2, 99)
        return self.time_series


# --- Start the app ---
if __name__ == "__main__":
    app = QApplication(sys.argv)            # Required for all PyQt apps
    gui = NMFGui(data_folder='C:/users/zwx/Documents/qcad 20250317 ndata1')        # Create your custom GUI
    gui.resize(1200, 900)                    # Optional: window size
    gui.show()                              # Show the GUI
    sys.exit(app.exec_())                   # Start event loop (blocking)