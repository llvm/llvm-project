// Override pure virtual function.
// We instead emit zero for the pure virtual function component. See PR43094 for
// details.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O1 -o - -emit-llvm -fhalf-no-semantic-interposition | FileCheck %s

// CHECK: @_ZTV1A.local = internal unnamed_addr constant { [4 x i32] } { [4 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1A.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32] }, ptr @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1A3barEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32] }, ptr @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, align 4

// CHECK: @_ZTV1B.local = internal unnamed_addr constant { [4 x i32] } { [4 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1B.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1B3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1B3barEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [4 x i32] }, ptr @_ZTV1B.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, align 4

// CHECK: @_ZTV1A ={{.*}} unnamed_addr alias { [4 x i32] }, ptr @_ZTV1A.local
// CHECK: @_ZTV1B ={{.*}} unnamed_addr alias { [4 x i32] }, ptr @_ZTV1B.local

// CHECK-NOT: declare void @__cxa_pure_virtual() unnamed_addr

class A {
public:
  virtual void foo() = 0;
  virtual void bar();
};

class B : public A {
public:
  void foo() override;
  void bar() override;
};

void A::bar() {}
void B::foo() {}
void B::bar() {}

void A_foo(A *a) {
  a->foo();
}
