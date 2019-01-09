#!usr/bin/python
# -*-coding:UTF-8-*-

from sklearn import datasets
import os
import numpy as np
import matplotlib.pyplot as plt

from models.Kmeans import biKMeansClassifier,KMeansClassifier

def test_biKmeans(data,k):
    clf = biKMeansClassifier(k)
    clf.fit(data)
    cents = clf._centroids
    labels = clf._labels
    sse = clf._sse
    colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90', '#845868']
    for i in range(k):
        index = np.nonzero(labels == i)[0]
        x0 = data[index, 0]
        x1 = data[index, 1]
        y_i = i
        for j in range(len(x0)):
            plt.text(x0[j], x1[j], str(y_i), color=colors[i],
                     fontdict={'weight': 'bold', 'size': 6})
        plt.scatter(cents[i, 0], cents[i, 1], marker='x', color=colors[i],
                    linewidths=7)

    plt.title("SSE={:.2f}".format(sse))
    plt.axis([4, 9, 1, 5])
    outname = "biKmeans_result" + str(k) + ".png"
    plt.savefig(os.path.join(os.getcwd(),'result',outname))
    # 显示结果
    plt.show()

def test_Kmeans(data,k):
    clf = KMeansClassifier(k)
    clf.fit(data)
    cents = clf._centroids
    labels = clf._labels
    sse = clf._sse
    colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90', '#845868']
    for i in range(k):
        index = np.nonzero(labels == i)[0]
        x0 = data[index, 0]
        x1 = data[index, 1]
        y_i = i
        for j in range(len(x0)):
            plt.text(x0[j], x1[j], str(y_i), color=colors[i],
                     fontdict={'weight': 'bold', 'size': 6})
        plt.scatter(cents[i, 0], cents[i, 1], marker='x', color=colors[i],
                    linewidths=7)

    plt.title("SSE={:.2f}".format(sse))
    plt.axis([4, 9, 1, 5])
    outname = "Kmeans_result" + str(k) + ".png"
    plt.savefig(os.path.join(os.getcwd(),'result',outname))
    # 显示结果
    plt.show()


if __name__=='__main__':
    # 测试两个入口函数
    iris = datasets.load_iris()  # 加载scikit的iris数据集
    data = iris.data
    k = 4
    test_Kmeans(data,k)
    test_biKmeans(data,k)
