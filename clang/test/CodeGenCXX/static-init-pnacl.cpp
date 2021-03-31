// RUN: %clang_cc1 -emit-llvm -triple=le32-unknown-nacl -o - %s | FileCheck %s

int f();

// Test that PNaCl uses the Itanium/x86 ABI in which the static
// variable's guard variable is tested via "load i8 and compare with
// zero" rather than the ARM ABI which uses "load i32 and test the
// bottom bit".
void g() {
  static int a = f();
}
// CHECK: [[LOAD:%.*]] = load atomic i8, i8* bitcast (i64* @_ZGVZ1gvE1a to i8*) acquire, align 8
// CHECK-NEXT: [[GUARD:%.*]] = icmp eq i8 [[LOAD]], 0
// CHECK-NEXT: br i1 [[GUARD]]
