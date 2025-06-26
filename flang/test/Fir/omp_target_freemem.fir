// RUN: %flang_fc1 -emit-llvm  %s -o - | FileCheck %s

// CHECK-LABEL: define void @omp_target_allocmem_array_of_nonchar(
// CHECK: call void @omp_target_free(ptr {{.*}}, i32 0)
func.func @omp_target_allocmem_array_of_nonchar() -> () {
  %device = arith.constant 0 : i32
  %1 = fir.omp_target_allocmem %device : i32, !fir.array<3x3xi32>
  fir.omp_target_freemem %device, %1 : i32, !fir.heap<!fir.array<3x3xi32>>
  return
}

// CHECK-LABEL: define void @omp_target_allocmem_array_of_char(
// CHECK: call void @omp_target_free(ptr {{.*}}, i32 0)
func.func @omp_target_allocmem_array_of_char() -> () {
  %device = arith.constant 0 : i32
  %1 = fir.omp_target_allocmem %device : i32, !fir.array<3x3x!fir.char<1,10>>
  fir.omp_target_freemem %device, %1 : i32, !fir.heap<!fir.array<3x3x!fir.char<1,10>>>
  return
}

// CHECK-LABEL: define void @omp_target_allocmem_array_of_dynchar(
// CHECK: call void @omp_target_free(ptr {{.*}}, i32 0)
func.func @omp_target_allocmem_array_of_dynchar(%l: i32) -> () {
  %device = arith.constant 0 : i32
  %1 = fir.omp_target_allocmem %device : i32, !fir.array<3x3x!fir.char<1,?>>(%l : i32)
  fir.omp_target_freemem %device, %1 : i32, !fir.heap<!fir.array<3x3x!fir.char<1,?>>>
  return
}
