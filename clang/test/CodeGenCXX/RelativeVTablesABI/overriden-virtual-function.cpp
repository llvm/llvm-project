// Check the layout of the vtable for a child class that inherits a virtual
// function but does override it.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O1 -S -o - -emit-llvm -fhalf-no-semantic-interposition | FileCheck %s

// CHECK: @_ZTV1B.local = private unnamed_addr constant { [4 x i32] } { [4 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1B.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1B3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1B3barEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, align 4

// CHECK: @_ZTV1B ={{.*}} unnamed_addr alias { [4 x i32] }, ptr @_ZTV1B.local

class A {
public:
  virtual void foo();
};

class B : public A {
public:
  void foo() override;
  virtual void bar();
};

void A::foo() {}
void B::foo() {}

void A_foo(A *a) {
  a->foo();
}

void B_foo(B *b) {
  b->foo();
}
