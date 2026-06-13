// RUN: fir-opt -flang-licm --split-input-file %s | FileCheck %s

// Test that canMoveOutOf for acc.parallel allows hoisting of operations
// with trivial type operands and results, and prevents hoisting of operations
// with non-trivial types out of the compute region.
//
// acc.loop LICM hoists both invariant fir.convert ops into acc.parallel.
// scf.for nested LICM then hoists only the trivial one out of acc.parallel.
//
// CHECK-LABEL:   func.func @test_parallel_canMoveOutOf(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<10xf32>>)
// CHECK:         %[[CVT_TRIVIAL:.*]] = fir.convert %{{.*}} : (i32) -> i64
// CHECK:         scf.for
// CHECK:           acc.parallel {
// CHECK:             fir.convert %[[ARG0]] : (!fir.ref<!fir.array<10xf32>>) -> memref<10xf32>
// CHECK:             acc.loop
func.func @test_parallel_canMoveOutOf(%arg0: !fir.ref<!fir.array<10xf32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %c5_i32 = arith.constant 5 : i32
  %cst = arith.constant 1.000000e+00 : f32
  scf.for %i = %c0 to %c10 step %c1 {
    acc.parallel {
      acc.loop control(%iv : i32) = (%c1_i32 : i32) to (%c10_i32 : i32) step (%c1_i32 : i32) {
        %cvt_trivial = fir.convert %c5_i32 : (i32) -> i64
        %cvt_ref = fir.convert %arg0 : (!fir.ref<!fir.array<10xf32>>) -> memref<10xf32>
        %idx = fir.convert %iv : (i32) -> index
        memref.store %cst, %cvt_ref[%idx] : memref<10xf32>
        acc.yield
      } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
      acc.yield
    }
  }
  return
}

// -----

// Same test for acc.kernels.
// CHECK-LABEL:   func.func @test_kernels_canMoveOutOf(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<10xf32>>)
// CHECK:         %[[CVT_TRIVIAL:.*]] = fir.convert %{{.*}} : (i32) -> i64
// CHECK:         scf.for
// CHECK:           acc.kernels {
// CHECK:             fir.convert %[[ARG0]] : (!fir.ref<!fir.array<10xf32>>) -> memref<10xf32>
// CHECK:             acc.loop
func.func @test_kernels_canMoveOutOf(%arg0: !fir.ref<!fir.array<10xf32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %c5_i32 = arith.constant 5 : i32
  %cst = arith.constant 1.000000e+00 : f32
  scf.for %i = %c0 to %c10 step %c1 {
    acc.kernels {
      acc.loop control(%iv : i32) = (%c1_i32 : i32) to (%c10_i32 : i32) step (%c1_i32 : i32) {
        %cvt_trivial = fir.convert %c5_i32 : (i32) -> i64
        %cvt_ref = fir.convert %arg0 : (!fir.ref<!fir.array<10xf32>>) -> memref<10xf32>
        %idx = fir.convert %iv : (i32) -> index
        memref.store %cst, %cvt_ref[%idx] : memref<10xf32>
        acc.yield
      } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
      acc.terminator
    }
  }
  return
}

// -----

// Same test for acc.serial.
// CHECK-LABEL:   func.func @test_serial_canMoveOutOf(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<10xf32>>)
// CHECK:         %[[CVT_TRIVIAL:.*]] = fir.convert %{{.*}} : (i32) -> i64
// CHECK:         scf.for
// CHECK:           acc.serial {
// CHECK:             fir.convert %[[ARG0]] : (!fir.ref<!fir.array<10xf32>>) -> memref<10xf32>
// CHECK:             acc.loop
func.func @test_serial_canMoveOutOf(%arg0: !fir.ref<!fir.array<10xf32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %c5_i32 = arith.constant 5 : i32
  %cst = arith.constant 1.000000e+00 : f32
  scf.for %i = %c0 to %c10 step %c1 {
    acc.serial {
      acc.loop control(%iv : i32) = (%c1_i32 : i32) to (%c10_i32 : i32) step (%c1_i32 : i32) {
        %cvt_trivial = fir.convert %c5_i32 : (i32) -> i64
        %cvt_ref = fir.convert %arg0 : (!fir.ref<!fir.array<10xf32>>) -> memref<10xf32>
        %idx = fir.convert %iv : (i32) -> index
        memref.store %cst, %cvt_ref[%idx] : memref<10xf32>
        acc.yield
      } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
      acc.yield
    }
  }
  return
}

// -----

// Test transitive canMoveOutOf: scf.if containing a fir.convert with
// non-trivial operand must NOT be hoisted out of acc.parallel as a whole,
// even though scf.if itself yields a trivial type (i64).
// CHECK-LABEL:   func.func @test_parallel_transitive_canMoveOutOf(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<10xf32>>,
// CHECK-SAME:      %[[COND:.*]]: i1)
// CHECK:         scf.for
// CHECK:           acc.parallel {
// scf.if is hoisted from acc.loop into acc.parallel, but NOT out of
// acc.parallel because nested fir.convert has non-trivial operand type:
// CHECK:             %[[ADDR:.*]] = scf.if %[[COND]] -> (i64)
// CHECK:               fir.convert %[[ARG0]]
// CHECK:             acc.loop
func.func @test_parallel_transitive_canMoveOutOf(%arg0: !fir.ref<!fir.array<10xf32>>, %cond: i1) {
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %cst = arith.constant 1.000000e+00 : f32
  scf.for %i = %c0 to %c10 step %c1 {
    acc.parallel {
      acc.loop control(%iv : i32) = (%c1_i32 : i32) to (%c10_i32 : i32) step (%c1_i32 : i32) {
        %addr = scf.if %cond -> (i64) {
          %ptr = fir.convert %arg0 : (!fir.ref<!fir.array<10xf32>>) -> i64
          scf.yield %ptr : i64
        } else {
          scf.yield %c0_i64 : i64
        }
        %ref = fir.convert %addr : (i64) -> !fir.ref<f32>
        fir.store %cst to %ref : !fir.ref<f32>
        acc.yield
      } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
      acc.yield
    }
  }
  return
}
