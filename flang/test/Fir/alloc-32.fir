// RUN: %flang_fc1 -triple i686 -emit-llvm  %s -o - | FileCheck %s
// REQUIRES: x86-registered-target

// This is a check for calling malloc using i32 when on a 32-bit target (only).
// It doesn't contain the comprehensive tests that alloc.fir has, and
// that file should be used to exercise most code paths.

module attributes {
    fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "i686"
} {

// CHECK-LABEL: define ptr @allocmem_scalar_nonchar(
// CHECK: call ptr @malloc(i32 4)
func.func @allocmem_scalar_nonchar() -> !fir.heap<i32> {
  %1 = fir.allocmem i32
  return %1 : !fir.heap<i32>
}

// CHECK-LABEL: define ptr @allocmem_scalar_dynchar(
// CHECK-SAME: i32 %[[len:.*]])
// CHECK: %[[mul1:.*]] = sext i32 %[[len]] to i64
// CHECK: %[[mul2:.*]] = mul i64 1, %[[mul1]]
// CHECK: %[[trunc:.*]] = trunc i64 %[[mul2]] to i32
// CHECK: call ptr @malloc(i32 %[[trunc]])
func.func @allocmem_scalar_dynchar(%l : i32) -> !fir.heap<!fir.char<1,?>> {
  %1 = fir.allocmem !fir.char<1,?>(%l : i32)
  return %1 : !fir.heap<!fir.char<1,?>>
}

}
