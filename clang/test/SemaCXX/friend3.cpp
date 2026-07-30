// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -emit-llvm %s -o - | FileCheck %s

namespace pr8852 {
void foo();
struct S {
  friend void foo() {}
};

void main() {
  foo();
}
// CHECK: define {{.*}} @_ZN6pr88523fooEv
}

namespace pr9518 {
template<typename T>
struct provide {
  friend T f() { return T(); }
};

void g() {
  void f();
  provide<void> p;
  f();
}
// CHECK: define {{.*}} @_ZN6pr95181fEv
}
