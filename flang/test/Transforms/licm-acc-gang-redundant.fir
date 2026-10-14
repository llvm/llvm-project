// RUN: fir-opt -flang-licm --split-input-file %s | FileCheck %s

// In a compute region from acc.parallel, code outside of the scf.parallel is
// executed redundantly by all gangs, so an invariant loop is not moved there.
// CHECK-LABEL: func.func @parallel_origin(
// CHECK:         acc.compute_region
// CHECK-NOT:       scf.for
// CHECK:           scf.parallel
// CHECK:             %[[PROD:.*]] = scf.for {{.*}} -> (f32) {
// CHECK:             arith.mulf %[[PROD]],
// CHECK:         } <{origin = "acc.parallel"}>
func.func @parallel_origin(%arg0: memref<?xf32>, %arg1: index) {
  acc.kernel_environment {
    acc.compute_region ins(%a = %arg0, %n = %arg1) : (memref<?xf32>, index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %one = arith.constant 1.0 : f32
      %f = arith.constant 0.9 : f32
      scf.parallel (%i) = (%c0) to (%n) step (%c1) {
        %prod = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %one) -> (f32) {
          %m = arith.mulf %acc, %f : f32
          scf.yield %m : f32
        }
        %x = memref.load %a[%i] : memref<?xf32>
        %y = arith.mulf %prod, %x : f32
        memref.store %y, %a[%i] : memref<?xf32>
        scf.reduce
      }
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  return
}

// -----

// A compute region from acc.kernels has no gang-redundant code: the loop is
// moved out.
// CHECK-LABEL: func.func @kernels_origin(
// CHECK:         acc.compute_region
// CHECK:           scf.for {{.*}} -> (f32) {
// CHECK:           scf.parallel
// CHECK:         } <{origin = "acc.kernels"}>
func.func @kernels_origin(%arg0: memref<?xf32>, %arg1: index) {
  acc.kernel_environment {
    acc.compute_region ins(%a = %arg0, %n = %arg1) : (memref<?xf32>, index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %one = arith.constant 1.0 : f32
      %f = arith.constant 0.9 : f32
      scf.parallel (%i) = (%c0) to (%n) step (%c1) {
        %prod = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %one) -> (f32) {
          %m = arith.mulf %acc, %f : f32
          scf.yield %m : f32
        }
        %x = memref.load %a[%i] : memref<?xf32>
        %y = arith.mulf %prod, %x : f32
        memref.store %y, %a[%i] : memref<?xf32>
        scf.reduce
      }
      acc.yield
    } <{origin = "acc.kernels"}>
  }
  return
}

// -----

// Code in the body of an outer scf.parallel may be executed redundantly at
// the worker or vector level, so a loop is not moved there from an inner
// scf.parallel either.
// CHECK-LABEL: func.func @into_outer_parallel(
// CHECK:         acc.compute_region
// CHECK:           scf.parallel
// CHECK-NOT:         scf.for
// CHECK:             scf.parallel
// CHECK:               scf.for {{.*}} -> (f32) {
// CHECK:         } <{origin = "acc.parallel"}>
func.func @into_outer_parallel(%arg0: memref<?x?xf32>, %arg1: index) {
  acc.kernel_environment {
    acc.compute_region ins(%a = %arg0, %n = %arg1) : (memref<?x?xf32>, index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %one = arith.constant 1.0 : f32
      scf.parallel (%j) = (%c0) to (%n) step (%c1) {
        %jf = arith.index_cast %j : index to i32
        %fj = arith.sitofp %jf : i32 to f32
        scf.parallel (%i) = (%c0) to (%n) step (%c1) {
          %prod = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %one) -> (f32) {
            %m = arith.mulf %acc, %fj : f32
            scf.yield %m : f32
          }
          %x = memref.load %a[%i, %j] : memref<?x?xf32>
          %y = arith.mulf %prod, %x : f32
          memref.store %y, %a[%i, %j] : memref<?x?xf32>
          scf.reduce
        }
        scf.reduce
      }
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  return
}
