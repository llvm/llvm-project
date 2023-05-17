// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck %s

template <class T> struct A { A(); };
struct B { A<int> x; };
void a() {   
  B b;
}

// CHECK: call {{.*}} @_ZN1BC1Ev
// CHECK: define linkonce_odr {{.*}} @_ZN1BC1Ev(ptr {{.*}}%this) unnamed_addr
// CHECK: call {{.*}} @_ZN1AIiEC1Ev
