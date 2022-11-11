// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -S -o - -emit-llvm -fsanitize=hwaddress | FileCheck %s

/// The usual vtable will have default visibility. In this case, the actual
/// vtable is hidden and the alias is made public. With hwasan enabled, we want
/// to ensure the local one self-referenced in the hidden vtable is not
/// hwasan-instrumented.
// CHECK-DAG: @_ZTV1A.local = private unnamed_addr constant { [3 x i32] } { [3 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1A.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1A3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1A.local, i32 0, i32 0, i32 2) to i64)) to i32)] }, no_sanitize_hwaddress, align 4
// CHECK-DAG: @_ZTV1A = unnamed_addr alias { [3 x i32] }, ptr @_ZTV1A.local
// CHECK-DAG: @_ZTI1A.rtti_proxy = hidden unnamed_addr constant ptr @_ZTI1A, no_sanitize_hwaddress, comdat

class A {
public:
  virtual void foo();
};

void A::foo() {}

void A_foo(A *a) {
  a->foo();
}

/// If the vtable happens to be hidden, then the alias is not needed. In this
/// case, the original vtable struct itself should be unsanitized.
// CHECK-DAG: @_ZTV1B = hidden unnamed_addr constant { [3 x i32] } { [3 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZTI1B.rtti_proxy to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1B, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1B3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV1B, i32 0, i32 0, i32 2) to i64)) to i32)] }, no_sanitize_hwaddress, align 4
// CHECK-DAG: @_ZTI1B.rtti_proxy = hidden unnamed_addr constant ptr @_ZTI1B, no_sanitize_hwaddress, comdat

class __attribute__((visibility("hidden"))) B {
public:
  virtual void foo();
};

void B::foo() {}

void B_foo(B *b) {
  b->foo();
}
