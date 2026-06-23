// RUN: %clang_cc1 -std=c++11 -ast-dump %s | FileCheck %s

void delfunc() = delete;
// CHECK: FunctionDecl{{.*}} <{{.*}}[[@LINE-1]]:1, col:23> col:6 delfunc 'void ()' delete

struct S {
  inline S();
};

inline S::S() = default;
// CHECK: CXXConstructorDecl{{.*}} <line:[[@LINE-1]]:1, col:23>{{.*}}S 'void ()' inline default

struct v {
  void f() = delete;
};
// CHECK: CXXMethodDecl{{.*}} <line:[[@LINE-2]]:3, col:19> col:8 f 'void ()' delete

class Truck {
  inline Truck();
};

inline Truck::Truck() = default;
// CHECK: CXXConstructorDecl{{.*}} <line:[[@LINE-1]]:1, col:31>{{.*}}Truck 'void ()' inline default

template <class T> void bubbleSort(T a[], int n) = delete;
// CHECK: FunctionTemplateDecl{{.*}} <line:[[@LINE-1]]:1, col:57> col:25 bubbleSort

template <typename T> class Array {
public:
    Array(T arr[], int s);
    void print() = delete;
};
// CHECK: CXXMethodDecl{{.*}} <line:[[@LINE-2]]:5, col:25> col:10 print 'void ()' delete
