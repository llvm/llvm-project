// Tests that LOCAL_INIT locality specifier is correctly converted to OpenMP
// for the host path. The key verification points are:
//
//   1. The omp.private op has type=private with NO copy_region (same as LOCAL).
//   2. The copy_region operations from fir.local are inlined at the top of
//      the omp.loop_nest body, giving per-iteration initialization.
//
// This ensures LOCAL_INIT's per-iteration semantics (each iteration sees the
// original value) rather than OpenMP firstprivate's per-thread semantics.

// RUN: fir-opt --omp-do-concurrent-conversion="map-to=host" %s | FileCheck %s

// A LOCAL_INIT localizer for f32. The copy_region loads from the original
// variable and stores to the private — this will be inlined per-iteration.
fir.local {type = local_init} @_QFtestEtmp_firstprivate_f32 : f32 copy {
^bb0(%arg0: !fir.ref<f32>, %arg1: !fir.ref<f32>):
  %0 = fir.load %arg0 : !fir.ref<f32>
  fir.store %0 to %arg1 : !fir.ref<f32>
  fir.yield(%arg1 : !fir.ref<f32>)
}

func.func @_QPtest(%arg0: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 3.140000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg1 dummy_scope %0 uniq_name("_QFtestEn") : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = fir.alloca f32 {bindc_name = "tmp", uniq_name = "_QFtestEtmp"}
  %3:2 = hlfir.declare %2 uniq_name("_QFtestEtmp") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %4 = fir.load %1#0 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> i64
  %6 = fir.convert %5 : (i64) -> index
  %7 = arith.cmpi sgt, %6, %c0 : index
  %8 = arith.select %7, %6, %c0 : index
  %9 = fir.shape %8 : (index) -> !fir.shape<1>
  %10:2 = hlfir.declare %arg0(%9) dummy_scope %0 uniq_name("_QFtestEa") : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
  fir.store %cst to %3#0 : !fir.ref<f32>
  %11 = fir.convert %c1_i32 : (i32) -> index
  %12 = fir.load %1#0 : !fir.ref<i32>
  %13 = fir.convert %12 : (i32) -> index

  fir.do_concurrent {
    %14 = fir.alloca i32 {bindc_name = "i"}
    %15:2 = hlfir.declare %14 uniq_name("_QFtestEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

    fir.do_concurrent.loop (%arg2) = (%11) to (%13) step (%c1)
      local(@_QFtestEtmp_firstprivate_f32 %3#0 -> %arg3 : !fir.ref<f32>) {
      %16 = fir.convert %arg2 : (index) -> i32
      fir.store %16 to %15#0 : !fir.ref<i32>
      %17:2 = hlfir.declare %arg3 uniq_name("_QFtestEtmp") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
      %18 = fir.load %17#0 : !fir.ref<f32>
      %19 = fir.load %15#0 : !fir.ref<i32>
      %20 = fir.convert %19 : (i32) -> f32
      %21 = arith.addf %18, %20 fastmath<contract> : f32
      hlfir.assign %21 to %17#0 : f32, !fir.ref<f32>
      %22 = fir.load %17#0 : !fir.ref<f32>
      %23 = fir.load %15#0 : !fir.ref<i32>
      %24 = fir.convert %23 : (i32) -> i64
      %25 = hlfir.designate %10#0 (%24) : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
      fir.store %22 to %25 : !fir.ref<f32>
    }
  }
  return
}

// Verify: omp.private has type=private and NO copy_region
// CHECK:      omp.private {type = private} @[[PRIVATIZER:.*_QFtestEtmp.*\.omp]] : f32

// Verify: omp.parallel + omp.wsloop + omp.loop_nest structure with
// private clause referencing the original variable.
// CHECK:      omp.parallel {
// CHECK:        omp.wsloop private(@[[PRIVATIZER]] %[[ORIG:.*]] -> %[[PRIV:.*]] : !fir.ref<f32>)
// CHECK:          omp.loop_nest

// Verify: per-iteration copy is inlined at the top of the loop_nest body.
// The first operations in the loop_nest should be loading from the original
// variable and storing to the private variable.
// CHECK:            %[[LOADED:.*]] = fir.load %[[ORIG]] : !fir.ref<f32>
// CHECK-NEXT:       fir.store %[[LOADED]] to %[[PRIV]] : !fir.ref<f32>
