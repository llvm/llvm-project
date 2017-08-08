// RUN: %clang_cc1 -std=c++1z %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s

template<typename T> struct A {
  A(T = 0);
  A(void*);
};

template<typename T> A(T*) -> A<long>;
A() -> A<int>;

// CHECK-LABEL: @_Z1fPi(
void f(int *p) {
  // CHECK: @_ZN1AIiEC
  A a{};

  // CHECK: @_ZN1AIlEC
  A b = p;

  // CHECK: @_ZN1AIxEC
  A c = 123LL;
}
