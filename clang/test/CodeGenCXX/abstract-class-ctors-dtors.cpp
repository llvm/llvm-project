// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// Check that we dont emit the complete constructor/destructor for this class.
struct A {
  virtual void f() = 0;
  A();
  ~A();
};

// CHECK-NOT: define void @_ZN1AC1Ev
// CHECK: define void @_ZN1AC2Ev
// CHECK: define void @_ZN1AD1Ev
// CHECK: define void @_ZN1AD2Ev
A::A() { }

A::~A() { }
