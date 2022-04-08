// RUN: %clang_cc1 -no-opaque-pointers %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// CHECK: ; ModuleID
struct A {
  template<typename T>
  A(T);
};

template<typename T> A::A(T) {}

struct B {
  template<typename T>
  B(T);
};

template<typename T> B::B(T) {}

// CHECK-LABEL: define weak_odr void @_ZN1BC2IiEET_(%struct.B* {{[^,]*}} %this, i32 noundef %0) unnamed_addr
// CHECK-LABEL: define weak_odr void @_ZN1BC1IiEET_(%struct.B* {{[^,]*}} %this, i32 noundef %0) unnamed_addr
template B::B(int);

template<typename T>
struct C {
  void f() {
    int a[] = { 1, 2, 3 };
  }
};

void f(C<int>& c) {
  c.f();
}
