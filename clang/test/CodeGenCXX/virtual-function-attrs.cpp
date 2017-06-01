// RUN: %clang_cc1 %s -triple %itanium_abi_triple -std=c++11 -emit-llvm -o - | FileCheck %s

class A {
  virtual void f();
  virtual void g();
  virtual ~A();
};

void A::f() {}

// CHECK: define [[CC:(x86_thiscallcc )?]]void @_ZN1A1fEv({{.*}}) unnamed_addr
// CHECK: declare [[CC]]void @_ZN1A1gEv({{.*}}) unnamed_addr
// CHECK: declare {{.*}} @_ZN1AD1Ev({{.*}}) unnamed_addr
// CHECK: declare [[CC]]void @_ZN1AD0Ev({{.*}}) unnamed_addr
