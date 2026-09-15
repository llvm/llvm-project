// RUN: %clang_cc1 -ast-dump %s | FileCheck %s

void delfunc() = delete;
// CHECK: FunctionDecl{{.*}} <{{.*}}[[@LINE-1]]:1, col:18> col:6 delfunc 'void ()' delete

struct S {
  inline S();
};

inline S::S() = default;
// CHECK: CXXConstructorDecl{{.*}} <line:[[@LINE-1]]:1, col:17>{{.*}}S 'void ()' inline default

struct v {
  void f() = delete;
};
// CHECK: CXXMethodDecl{{.*}} <line:[[@LINE-2]]:3, col:14> col:8 f 'void ()' delete

template <class T> void bubbleSort(T a[], int n) = delete;
// CHECK: FunctionTemplateDecl{{.*}} <line:[[@LINE-1]]:1, col:52> col:25 bubbleSort

template <typename T> class Array {
public:
    Array(T arr[], int s);
    void print() = delete;
};
// CHECK: CXXMethodDecl{{.*}} <line:[[@LINE-2]]:5, col:20> col:10 print 'void ()' delete

void delfunc2() = delete("reason");
// CHECK: FunctionDecl{{.*}} <{{.*}}[[@LINE-1]]:1, col:34> col:6 delfunc2 'void ()' delete

struct w {
  void g() = delete("reason");
};
// CHECK: CXXMethodDecl{{.*}} <line:[[@LINE-2]]:3, col:29> col:8 g 'void ()' delete
