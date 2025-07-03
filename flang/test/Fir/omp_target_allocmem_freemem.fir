// RUN: %flang_fc1 -emit-llvm  %s -o - | FileCheck %s

// CHECK-LABEL: define void @omp_target_allocmem_array_of_nonchar(
// CHECK: call ptr @omp_target_alloc(i64 36, i32 0)
// CHECK: call void @omp_target_free(ptr {{.*}}, i32 0)
func.func @omp_target_allocmem_array_of_nonchar() -> () {
  %device = arith.constant 0 : i32
  %1 = omp.target_allocmem %device : i32, !fir.array<3x3xi32>
  omp.target_freemem %device, %1 : i32, i64
  return
}

// CHECK-LABEL: define void @omp_target_allocmem_array_of_char(
// CHECK: call ptr @omp_target_alloc(i64 90, i32 0)
// CHECK: call void @omp_target_free(ptr {{.*}}, i32 0)
func.func @omp_target_allocmem_array_of_char() -> () {
  %device = arith.constant 0 : i32
  %1 = omp.target_allocmem %device : i32, !fir.array<3x3x!fir.char<1,10>>
  omp.target_freemem %device, %1 : i32, i64
  return
}

// CHECK-LABEL: define void @omp_target_allocmem_array_of_dynchar(
// CHECK-SAME: i32 %[[len:.*]])
// CHECK: %[[mul1:.*]] = sext i32 %[[len]] to i64
// CHECK: %[[mul2:.*]] = mul i64 9, %[[mul1]]
// CHECK: call ptr @omp_target_alloc(i64 %[[mul2]], i32 0)
func.func @omp_target_allocmem_array_of_dynchar(%l: i32) -> () {
  %device = arith.constant 0 : i32
  %1 = omp.target_allocmem %device : i32, !fir.array<3x3x!fir.char<1,?>>(%l : i32)
  omp.target_freemem %device, %1 : i32, i64
  return
}
