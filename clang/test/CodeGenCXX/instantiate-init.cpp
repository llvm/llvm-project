// RUN: %clang_cc1 -triple x86_64-linux -std=c++14 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux -std=c++17 %s -emit-llvm -o - | FileCheck %s

namespace std {
  template<typename T> class initializer_list {
    const T *data;
    __SIZE_TYPE__ size;

  public:
    initializer_list();
  };
}

namespace ParenBraceInitList {
  struct Vector {
    Vector(std::initializer_list<int>);
    ~Vector();
  };

  struct Base { Base(Vector) {} };

  // CHECK: define {{.*}}18ParenBraceInitList1fILi0EE
  template<int> void f() {
    // CHECK: call {{.*}}18ParenBraceInitList6VectorC1
    // CHECK: call {{.*}}18ParenBraceInitList6VectorD1
    Base({0});
  }
  template void f<0>();
}
