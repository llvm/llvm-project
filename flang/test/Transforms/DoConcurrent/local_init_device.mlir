// Tests that LOCAL_INIT locality specifier is correctly converted to OpenMP
// for the device path. The key verification points are:
//
//   1. The omp.private op has type=private with NO copy_region (same as LOCAL).
//   2. The private variable is attached to omp.parallel (not omp.wsloop) on the
//      device path.
//   3. The copy_region operations from fir.local are inlined at the top of
//      the omp.loop_nest body, giving per-iteration initialization.
//
// This ensures LOCAL_INIT's per-iteration semantics (each iteration sees the
// original value) rather than OpenMP firstprivate's per-thread semantics.

// RUN: fir-opt --omp-do-concurrent-conversion="map-to=device" %s | FileCheck %s

// A LOCAL_INIT localizer for f32. The copy_region loads from the original
// variable and stores to the private.
fir.local {type = local_init} @_QFtestEtmp_firstprivate_f32 : f32 copy {
^bb0(%arg0: !fir.ref<f32>, %arg1: !fir.ref<f32>):
  %0 = fir.load %arg0 : !fir.ref<f32>
  fir.store %0 to %arg1 : !fir.ref<f32>
  fir.yield(%arg1 : !fir.ref<f32>)
}

func.func @_QPtest() {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant 3.140000e+00 : f32

  %0 = fir.alloca f32 {bindc_name = "tmp", uniq_name = "_QFtestEtmp"}
  %1 = fir.declare %0 uniq_name("_QFtestEtmp") : (!fir.ref<f32>) -> !fir.ref<f32>
  fir.store %cst to %1 : !fir.ref<f32>

  %2 = fir.address_of(@_QFEa) : !fir.ref<!fir.array<10xf32>>
  %3 = fir.shape %c10 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %2(%3) uniq_name("_QFtestEa") : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)

  fir.do_concurrent {
    %5 = fir.alloca i32 {bindc_name = "i"}
    %6:2 = hlfir.declare %5 uniq_name("_QFtestEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

    fir.do_concurrent.loop (%arg0) = (%c1) to (%c10) step (%c1)
      local(@_QFtestEtmp_firstprivate_f32 %1 -> %arg1 : !fir.ref<f32>) {
      %7 = fir.convert %arg0 : (index) -> i32
      fir.store %7 to %6#0 : !fir.ref<i32>
      %8 = fir.declare %arg1 uniq_name("_QFtestEtmp") : (!fir.ref<f32>) -> !fir.ref<f32>
      %9 = fir.load %8 : !fir.ref<f32>
      %10 = fir.load %6#0 : !fir.ref<i32>
      %11 = fir.convert %10 : (i32) -> f32
      %12 = arith.addf %9, %11 fastmath<contract> : f32
      hlfir.assign %12 to %8 : f32, !fir.ref<f32>
    }
  }
  return
}

fir.global @_QFEa : !fir.array<10xf32>

// Verify: omp.private has type=private and NO copy_region
// CHECK:      omp.private {type = private} @[[PRIVATIZER:.*_QFtestEtmp.*\.omp]] : f32

// Verify: full device nesting structure with private on omp.parallel
// CHECK:      omp.target
// CHECK:        omp.teams {
// CHECK:          omp.parallel private(@[[PRIVATIZER]] %[[ORIG:.*]] -> %[[PRIV:.*]] : !fir.ref<f32>) {
// CHECK:            omp.distribute {
// CHECK:              omp.wsloop {
// CHECK:                omp.loop_nest

// Verify: per-iteration copy is inlined at the top of the loop_nest body
// CHECK:                  %[[LOADED:.*]] = fir.load %[[ORIG]] : !fir.ref<f32>
// CHECK-NEXT:             fir.store %[[LOADED]] to %[[PRIV]] : !fir.ref<f32>
