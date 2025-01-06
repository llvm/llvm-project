// Check the layout of the vtable for a child class that inherits a virtual
// function but does not override it.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O1 -o - -emit-llvm -fhalf-no-semantic-interposition | FileCheck %s

class A {
public:
  virtual void foo();
};

// The VTable for B should look similar to the vtable for A but the component for foo() should point to A::foo() and the component for bar() should point to B::bar().
// CHECK: @_ZTV1B.local = internal unnamed_addr constant { [4 x i32] } { [4 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1B.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1A3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1B3barEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, align 4
// CHECK: @_ZTV1B ={{.*}} unnamed_addr alias { [4 x i32] }, ptr @_ZTV1B.local

class B : public A {
public:
  virtual void bar();
};

void A::foo() {}
void B::bar() {}

void A_foo(A *a) {
  a->foo();
}

void B_foo(B *b) {
  b->foo();
}
