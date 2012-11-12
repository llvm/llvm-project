// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

struct A;

struct B {
  virtual void f();
  virtual A g();
};

void B::f() { }

// CHECK: define i32 @_ZN1D1gEv(%struct.D* %this)
// CHECK: declare void @_ZN1B1gEv()

struct C;

struct D {
  virtual void f();
  virtual C g();
};

void D::f() { }

struct C {
  int a;
};

C D::g() {
  return C();
}
