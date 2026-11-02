// RUN: fir-opt -flang-licm --split-input-file %s | FileCheck %s

// A loop-like operation is never hoisted out of the loop it is nested in, even
// if it is invariant and free of side effects. Hoisting it changes how it is
// scheduled relative to the iterations of the enclosing loop.

// CHECK-LABEL:   func.func @inner_loop_stays(
// CHECK-NOT:       scf.for
// CHECK:           scf.parallel
// CHECK:             scf.for
// CHECK:             memref.store
func.func @inner_loop_stays(%arg0: memref<?xf32>, %arg1: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.899999976 : f32
  scf.parallel (%i) = (%c0) to (%arg1) step (%c1) {
    %s = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %cst) -> (f32) {
      %m = arith.mulf %acc, %cst_0 : f32
      scf.yield %m : f32
    }
    memref.store %s, %arg0[%i] : memref<?xf32>
    scf.reduce
  }
  return
}

// -----

// The same, with the inner loop inside a conditional region of the outer loop.

// CHECK-LABEL:   func.func @inner_loop_in_if_stays(
// CHECK-NOT:       scf.for
// CHECK:           scf.parallel
// CHECK:             scf.if
// CHECK:               scf.for
func.func @inner_loop_in_if_stays(%arg0: memref<?xf32>, %arg1: index, %arg2: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.899999976 : f32
  scf.parallel (%i) = (%c0) to (%arg1) step (%c1) {
    scf.if %arg2 {
      %s = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %cst) -> (f32) {
        %m = arith.mulf %acc, %cst_0 : f32
        scf.yield %m : f32
      }
      memref.store %s, %arg0[%i] : memref<?xf32>
    }
    scf.reduce
  }
  return
}
