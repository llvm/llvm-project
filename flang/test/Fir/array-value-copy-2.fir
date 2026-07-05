// RUN: fir-opt --array-value-copy --cfg-conversion %s | FileCheck %s
// RUN: fir-opt --array-value-copy="optimize-conflicts=true" --cfg-conversion %s | FileCheck %s

// CHECK-LABEL: func @_QPslice1(
// CHECK-NOT: fir.allocmem
// CHECK: ^bb{{[0-9]+}}(%{{[0-9]+}}: index,
// CHECK-NOT: ^bb{{[0-9]+}}(%{{[0-9]+}}: index,
// CHECK: return
func.func @_QPslice1(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: !fir.ref<i32>, %arg2: !fir.ref<i32>) {
  %c1_i64 = arith.constant 1 : i64
  %0 = fir.convert %c1_i64 : (i64) -> index
  %1 = fir.load %arg1 : !fir.ref<i32>
  %2 = fir.convert %1 : (i32) -> i64
  %3 = fir.convert %2 : (i64) -> index
  %c0 = arith.constant 0 : index
  %4 = arith.subi %3, %0 : index
  %5 = arith.addi %4, %0 : index
  %6 = arith.divsi %5, %0 : index
  %7 = arith.cmpi sgt, %6, %c0 : index
  %8 = arith.select %7, %6, %c0 : index
  %9 = fir.slice %0, %3, %0 : (index, index, index) -> !fir.slice<1>
  %10 = fir.array_load %arg0 [%9] : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>) -> !fir.array<?xf32>
  %c1_i32 = arith.constant 1 : i32
  %11 = arith.addi %1, %c1_i32 : i32
  %12 = fir.convert %11 : (i32) -> i64
  %13 = fir.convert %12 : (i64) -> index
  %14 = fir.load %arg2 : !fir.ref<i32>
  %15 = fir.convert %14 : (i32) -> i64
  %16 = fir.convert %15 : (i64) -> index
  %17 = fir.slice %13, %16, %0 : (index, index, index) -> !fir.slice<1>
  %18 = fir.array_load %arg0 [%17] : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>) -> !fir.array<?xf32>
  %c1 = arith.constant 1 : index
  %19 = arith.subi %8, %c1 : index
  %20 = fir.do_loop %arg3 = %c0 to %19 step %c1 unordered iter_args(%arg4 = %10) -> (!fir.array<?xf32>) {
    %21 = fir.array_fetch %18, %arg3 : (!fir.array<?xf32>, index) -> f32
    %22 = fir.array_update %arg4, %21, %arg3 : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
    fir.result %22 : !fir.array<?xf32>
  }
  fir.array_merge_store %10, %20 to %arg0[%9] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.box<!fir.array<?xf32>>, !fir.slice<1>
  return
}

// CHECK-LABEL: func @_QPslice2(
// CHECK: fir.allocmem
// CHECK: ^bb{{[0-9]+}}(%{{[0-9]+}}: index,
// CHECK: ^bb{{[0-9]+}}(%{{[0-9]+}}: index,
// CHECK: fir.freemem
// CHECK: return
func.func @_QPslice2(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: !fir.ref<i32>, %arg2: !fir.ref<i32>) {
  %c1_i64 = arith.constant 1 : i64
  %0 = fir.convert %c1_i64 : (i64) -> index
  %c1_i64_0 = arith.constant 1 : i64
  %1 = fir.convert %c1_i64_0 : (i64) -> index
  %2 = fir.load %arg1 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> i64
  %4 = fir.convert %3 : (i64) -> index
  %c0 = arith.constant 0 : index
  %5 = arith.subi %4, %0 : index
  %6 = arith.addi %5, %1 : index
  %7 = arith.divsi %6, %1 : index
  %8 = arith.cmpi sgt, %7, %c0 : index
  %9 = arith.select %8, %7, %c0 : index
  %10 = fir.slice %0, %4, %1 : (index, index, index) -> !fir.slice<1>
  %11 = fir.array_load %arg0 [%10] : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>) -> !fir.array<?xf32>
  %12 = fir.load %arg1 : !fir.ref<i32>
  %c4_i32 = arith.constant 4 : i32
  %13 = arith.subi %12, %c4_i32 : i32
  %14 = fir.convert %13 : (i32) -> i64
  %15 = fir.convert %14 : (i64) -> index
  %c1_i64_1 = arith.constant 1 : i64
  %16 = fir.convert %c1_i64_1 : (i64) -> index
  %17 = fir.load %arg2 : !fir.ref<i32>
  %18 = fir.convert %17 : (i32) -> i64
  %19 = fir.convert %18 : (i64) -> index
  %20 = fir.slice %15, %19, %16 : (index, index, index) -> !fir.slice<1>
  %21 = fir.array_load %arg0 [%20] : (!fir.box<!fir.array<?xf32>>, !fir.slice<1>) -> !fir.array<?xf32>
  %c1 = arith.constant 1 : index
  %c0_2 = arith.constant 0 : index
  %22 = arith.subi %9, %c1 : index
  %23 = fir.do_loop %arg3 = %c0_2 to %22 step %c1 unordered iter_args(%arg4 = %11) -> (!fir.array<?xf32>) {
    %24 = fir.array_fetch %21, %arg3 : (!fir.array<?xf32>, index) -> f32
    %25 = fir.array_update %arg4, %24, %arg3 : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
    fir.result %25 : !fir.array<?xf32>
  }
  fir.array_merge_store %11, %23 to %arg0[%10] : !fir.array<?xf32>, !fir.array<?xf32>, !fir.box<!fir.array<?xf32>>, !fir.slice<1>
  return
}
