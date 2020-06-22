// Check that we do not emit a stub for a virtual function if it is already
// known to be in the same linkage unit as the vtable.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -S -o - -emit-llvm -fexperimental-relative-c++-abi-vtables | FileCheck %s --check-prefixes=CHECK,NO-OPT
// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -S -o - -emit-llvm -fexperimental-relative-c++-abi-vtables -O3 | FileCheck %s --check-prefixes=CHECK,OPT

// The vtable offset is relative to _ZN1A3fooEv instead of a stub. We can do
// this since hidden functions are implicitly dso_local.
// CHECK: @_ZTV1A.local = private unnamed_addr constant { [5 x i32] } { [5 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8* }** @_ZTI1A.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [5 x i32] }, { [5 x i32] }* @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.A*)* @_ZN1A3fooEv to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [5 x i32] }, { [5 x i32] }* @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (%class.A* (%class.A*)* @_ZN1AD1Ev.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [5 x i32] }, { [5 x i32] }* @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.A*)* @_ZN1AD0Ev.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [5 x i32] }, { [5 x i32] }* @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, align 4

// Despite the complete object destructor being hidden, we should still emit a
// stub for it because it's possible, when optimizations are enabled, for the
// dtor to be "devirtualized" into the destructor for a parent class if the
// child class doesn't implement its own dtor. For complete destructors, we
// always emit and use a stub.
// @_ZTV1B = hidden unnamed_addr constant { [5 x i32] } { [5 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, i8*, i8* }** @_ZTI1B.rtti_proxy to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [5 x i32] }, { [5 x i32] }* @_ZTV1B, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.B*)* @_ZN1B3fooEv to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [5 x i32] }, { [5 x i32] }* @_ZTV1B, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (%class.B* (%class.B*)* @_ZN1BD1Ev.stub to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [5 x i32] }, { [5 x i32] }* @_ZTV1B, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (void (%class.B*)* @_ZN1BD0Ev to i64), i64 ptrtoint (i32* getelementptr inbounds ({ [5 x i32] }, { [5 x i32] }* @_ZTV1B, i32 0, i32 0, i32 2) to i64)) to i32)] }, align 4

// CHECK: @_ZTV1A = unnamed_addr alias { [5 x i32] }, { [5 x i32] }* @_ZTV1A.local

// CHECK: @_ZN1A3fooEv
// CHECK-NOT: @_ZN1A3fooEv.stub

// The complete object destructor is hidden.
// NO-OPT: define linkonce_odr hidden %class.B* @_ZN1BD1Ev
// OPT-NOT: @_ZN1BD1Ev
// CHECK: @_ZN1BD1Ev.stub

#define HIDDEN __attribute__((visibility("hidden")))

class A {
public:
  HIDDEN virtual void foo();
  virtual ~A();
};

class HIDDEN B : public A {
public:
  virtual void foo();
};

void A::foo() {}

void A_foo(A *a) {
  a->foo();
}

void B::foo() {
  A::foo();
}
