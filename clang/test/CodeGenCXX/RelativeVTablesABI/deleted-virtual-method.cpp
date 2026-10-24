// Check the layout of the vtable for a class with a deleted virtual function.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -O1 -o - -emit-llvm -fhalf-no-semantic-interposition | FileCheck %s

// CHECK: @_ZTV1A.local = internal constant { [4 x i32] } { [4 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1A.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds (i8, ptr @_ZTV1A.local, i64 8) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @__cxa_deleted_virtual to i64), i64 ptrtoint (ptr getelementptr inbounds (i8, ptr @_ZTV1A.local, i64 8) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1A3barEv to i64), i64 ptrtoint (ptr getelementptr inbounds (i8, ptr @_ZTV1A.local, i64 8) to i64)) to i32)] }, align 4

// CHECK: @_ZTV1A ={{.*}}alias { [4 x i32] }, ptr @_ZTV1A.local

// CHECK: declare void @__cxa_deleted_virtual() unnamed_addr

class A {
public:
  virtual void foo() = delete;
  virtual void bar();
};

void A::bar() {}
