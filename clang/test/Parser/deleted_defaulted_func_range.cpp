// RUN: %clang_cc1 -fsyntax-only -ast-dump %s | FileCheck %s

// CHECK: FunctionDecl {{.*}} <{{.*}}:4:1, col:17> {{.*}}
void f() = delete;


struct S {
  inline S();
  ~S();
};

//CHECK: CXXConstructorDecl {{.*}} <{{.*}}:13:1, col:23> {{.*}}
inline S::S() = default;
//CHECK: CXXDestructorDecl {{.*}} <{{.*}}:15:1, col:17> {{.*}}
S::~S() = default;

template <typename T>
class U {
  U();
  ~U();
};

//CHECK: CXXConstructorDecl {{.*}} <{{.*}}:24:1, line:25:19> {{.*}}
template <typename T>
U<T>::U() = default;
//CHECK: CXXDestructorDecl {{.*}} <{{.*}}:27:1, line:28:20> {{.*}}
template <typename T>
U<T>::~U() = default;