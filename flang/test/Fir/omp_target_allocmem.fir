// RUN: %flang_fc1 -emit-llvm  %s -o - | FileCheck %s

// CHECK-LABEL: define ptr @omp_target_allocmem_array_of_nonchar(
// CHECK: call ptr @omp_target_alloc(i64 36, i32 0)
func.func @omp_target_allocmem_array_of_nonchar() -> !fir.heap<!fir.array<3x3xi32>> {
  %device = arith.constant 0 : i32
  %1 = fir.omp_target_allocmem %device : i32, !fir.array<3x3xi32>
  return %1 : !fir.heap<!fir.array<3x3xi32>>
}

// CHECK-LABEL: define ptr @omp_target_allocmem_array_of_char(
// CHECK: call ptr @omp_target_alloc(i64 90, i32 0)
func.func @omp_target_allocmem_array_of_char() -> !fir.heap<!fir.array<3x3x!fir.char<1,10>>> {
  %device = arith.constant 0 : i32
  %1 = fir.omp_target_allocmem %device : i32, !fir.array<3x3x!fir.char<1,10>>
  return %1 : !fir.heap<!fir.array<3x3x!fir.char<1,10>>>
}

// CHECK-LABEL: define ptr @omp_target_allocmem_array_of_dynchar(
// CHECK-SAME: i32 %[[len:.*]])
// CHECK: %[[mul1:.*]] = sext i32 %[[len]] to i64
// CHECK: %[[mul2:.*]] = mul i64 9, %[[mul1]]
// CHECK: call ptr @omp_target_alloc(i64 %[[mul2]], i32 0)
func.func @omp_target_allocmem_array_of_dynchar(%l: i32) -> !fir.heap<!fir.array<3x3x!fir.char<1,?>>> {
  %device = arith.constant 0 : i32
  %1 = fir.omp_target_allocmem %device : i32, !fir.array<3x3x!fir.char<1,?>>(%l : i32)
  return %1 : !fir.heap<!fir.array<3x3x!fir.char<1,?>>>
}
