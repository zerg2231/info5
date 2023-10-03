import cv2
import numpy as np
import sys

from matplotlib import pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(253, 153)
        MainWindow.setMinimumSize(QtCore.QSize(253, 153))
        MainWindow.setMaximumSize(QtCore.QSize(253, 153))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 141, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 40, 141, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 70, 141, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 100, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(160, 10, 51, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(160, 40, 51, 20))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(160, 70, 51, 20))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(80, 100, 81, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.button_click)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Стиль"))
        self.label.setText(_translate("MainWindow", "Количество эталонов:"))
        self.label_2.setText(_translate("MainWindow", "Количество тестовых:"))
        self.label_3.setText(_translate("MainWindow", "Скорость отображения:"))
        self.label_4.setText(_translate("MainWindow", "Результат:"))
        self.pushButton.setText(_translate("MainWindow", "Вывод"))
        
    def button_click(self):
        Program(int(self.lineEdit.text()), int(self.lineEdit_2.text()), float(self.lineEdit_3.text())).show_result()

class Image():
    def __init__(self, i, j, tp):
        self.tp = tp
        self.image = cv2.imread(f'paintings\style{i + 1}\{j + 1}.jpg')
        self.image_gray = cv2.imread(f'paintings\style{i + 1}\{j + 1}.jpg', 0)
        
    def get_image(self):
        if self.tp == 'normal':
            return self.image
        
        elif self.tp == 'normal_r':
            return cv2.resize(self.image, (500, 500))
        
        elif self.tp == 'gray_r':
            return cv2.resize(self.image_gray, (500, 500))
    
class Program():
    def __init__(self, count, amount, speed):
        self.count = count
        self.amount = amount
        self.speed = speed
        self.standards = [[] for i in range(5)]
        self.tests = [[] for i in range(5)]
        self.get_dataset()
    
    def get_dataset(self):
        for i in range(5):
            for j in range(self.count):
                standard = Image(i, j, 'gray_r').get_image()
                self.standards[i].append(standard)
                
            for j in range(self.count, self.amount + self.count):
                test = Image(i, j, 'gray_r').get_image()
                self.tests[i].append(test)
    
    def similarity(self, image_1, image_2):
        dist = np.linalg.norm(image_1 - image_2)

        return dist
    
    def hog(self, image):
        hog = cv2.HOGDescriptor()
        result = hog.compute(image)
        
        return result
    
    def hl(self, image):
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines_list =[]
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                minLineLength=1, maxLineGap=10)
        
        for points in lines:
            x1, y1, x2, y2 = points[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            lines_list.append([(x1, y1), (x2, y2)])
        
        return image
    
    def gf(self, image):
        g_kernel = cv2.getGaborKernel((10, 10), 8.0, np.pi/4, 10.0, 0.5, 0, 
                                      ktype=cv2.CV_32F)
        filtered_image = cv2.filter2D(image, cv2.CV_8UC3, g_kernel)
        
        return filtered_image
    
    def compare_images(self, tp):
        result = [[] for i in range(5)]
        
        for i in range(5):
            for j in range(len(self.tests[0])):
                test = self.tests[i][j]
                compared_1 = []
                
                for k in range(5):
                    compared_2 = []
                    
                    for l in range(len(self.standards[0])):
                        if tp == 'hog':
                            image_1 = self.hog(self.standards[k][l])
                            image_2 = self.hog(test)
                            
                        elif tp == 'hl':
                            image_1 = self.hl(self.standards[k][l])
                            image_2 = self.hl(test)
                            
                        elif tp == 'gf':
                            image_1 = self.gf(self.standards[k][l])
                            image_2 = self.gf(test)
                        
                        sim = self.similarity(image_1, image_2)
                        compared_2.append(sim)
                    
                    compared_1.append(np.max(compared_2))
                
                result[i].append([j, np.min(compared_1),
                                  np.argmin(compared_1)])
        
        return result
    
    def get_results(self):
        self.results = []
        
        for method in 'hog', 'hl', 'gf':
            self.results.append(self.compare_images(method))
        
    def get_accuracy(self):
        self.accuracy = [[] for i in range(3)]
        
        for i in range(len(self.results)):
            trues = 0
            alls = 0
    
            for j in range(5):
                for k in range(len(self.tests[0])):
                    alls += 1
        
                    if self.results[i][j][k][2] == j:
                        trues += 1
        
                    self.accuracy[i].append((trues / alls) * 100)
    
    def get_style(self, i):
        if i == 0:
            return 'импрессионизм'
        
        elif i == 1:
            return 'постимпрессионизм'
        
        elif i == 2:
            return 'кубизм'
        
        elif i == 3:
            return 'супрематизм'
        
        elif i == 4:
            return 'сюрреализм'
                        
    def show_result(self):
        self.get_results()
        self.get_accuracy()
        
        fig = plt.figure('Результат', figsize=(18, 9))
        
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(3, 2, 2)
        ax3 = fig.add_subplot(3, 2, 4)
        ax4 = fig.add_subplot(3, 2, 6)
        
        
        y = [[] for x in range(3)]
        x = []
        
        k = 0
        for i in range(5):
            for j in range(len(self.tests[0])):
                y[0].append(self.accuracy[0][k])
                y[1].append(self.accuracy[1][k])
                y[2].append(self.accuracy[2][k])
                x.append(k + 1)
                
                ax1.cla()
                ax1.imshow(cv2.cvtColor(Image(i, j, 'normal').get_image(), 
                                        cv2.COLOR_BGR2RGB))
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.set_title('Стиль: ' + str(self.get_style(i)))
                
                style_1 = self.get_style(self.results[0][i][j][2]) 
                style_2 = self.get_style(self.results[1][i][j][2]) 
                style_3 = self.get_style(self.results[2][i][j][2]) 
                
                ax1.set_xlabel('HOG: ' + str(style_1) + 
                               '\nHoughLines: ' + str(style_2) + 
                               '\nGaborFilter: ' + str(style_3))
                
                ax2.cla()
                ax2.plot(x, y[0])
                ax2.set_title('HOG')
                ax2.set_xlabel('Кол-во изображений')
                ax2.set_ylabel('Точность (%)')
                ax2.set_yticks(np.arange(0, 110, 10))
                
                ax3.cla()
                ax3.plot(x, y[1])
                ax3.set_title('HoughLines')
                ax3.set_xlabel('Кол-во изображений')
                ax3.set_ylabel('Точность (%)')
                ax3.set_yticks(np.arange(0, 110, 10))
                
                ax4.cla()
                ax4.plot(x, y[2])
                ax4.set_title('GaborFilter')
                ax4.set_xlabel('Кол-во изображений')
                ax4.set_ylabel('Точность (%)')
                ax4.set_yticks(np.arange(0, 110, 10))
                
                plt.subplots_adjust(wspace=0.5, hspace=0.5,
                                    top=0.95, bottom=0.1,
                                    left=0.05, right=0.95)
                plt.show()
                plt.pause(self.speed)
                
                k += 1

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.show()
sys.exit(app.exec_())