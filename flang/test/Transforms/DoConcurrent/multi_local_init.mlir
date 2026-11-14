// Tests two LOCAL_INIT variables in the same do concurrent loop. Verifies:
//   1. Two omp.private ops (one for each variable)
//   2. Both inlined copies appear at top of omp.loop_nest body
//   3. Each copy references the correct original and private variables

// RUN: fir-opt --omp-do-concurrent-conversion="map-to=host" %s | FileCheck %s

fir.local {type = local_init} @_QFtestEx_firstprivate_f32 : f32 copy {
^bb0(%arg0: !fir.ref<f32>, %arg1: !fir.ref<f32>):
  %0 = fir.load %arg0 : !fir.ref<f32>
  fir.store %0 to %arg1 : !fir.ref<f32>
  fir.yield(%arg1 : !fir.ref<f32>)
}

fir.local {type = local_init} @_QFtestEy_firstprivate_f32 : f32 copy {
^bb0(%arg0: !fir.ref<f32>, %arg1: !fir.ref<f32>):
  %0 = fir.load %arg0 : !fir.ref<f32>
  fir.store %0 to %arg1 : !fir.ref<f32>
  fir.yield(%arg1 : !fir.ref<f32>)
}

func.func @_QPmulti_local_init() {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %0 = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFtestEx"}
  %1:2 = hlfir.declare %0 uniq_name("_QFtestEx") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)

  %2 = fir.alloca f32 {bindc_name = "y", uniq_name = "_QFtestEy"}
  %3:2 = hlfir.declare %2 uniq_name("_QFtestEy") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)

  %cst1 = arith.constant 1.000000e+00 : f32
  fir.store %cst1 to %1#0 : !fir.ref<f32>
  %cst2 = arith.constant 2.000000e+00 : f32
  fir.store %cst2 to %3#0 : !fir.ref<f32>

  fir.do_concurrent {
    %4 = fir.alloca i32 {bindc_name = "i"}
    %5:2 = hlfir.declare %4 uniq_name("_QFtestEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

    fir.do_concurrent.loop (%arg0) = (%c1) to (%c10) step (%c1)
      local(@_QFtestEx_firstprivate_f32 %1#0 -> %arg1, @_QFtestEy_firstprivate_f32 %3#0 -> %arg2 : !fir.ref<f32>, !fir.ref<f32>) {
      %6 = fir.convert %arg0 : (index) -> i32
      fir.store %6 to %5#0 : !fir.ref<i32>
      %7:2 = hlfir.declare %arg1 uniq_name("_QFtestEx") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
      %8:2 = hlfir.declare %arg2 uniq_name("_QFtestEy") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
      %9 = fir.load %7#0 : !fir.ref<f32>
      %10 = fir.load %8#0 : !fir.ref<f32>
      %11 = arith.addf %9, %10 fastmath<contract> : f32
      hlfir.assign %11 to %7#0 : f32, !fir.ref<f32>
    }
  }
  return
}

// Verify: two omp.private ops, one for each variable
// CHECK-DAG: omp.private {type = private} @[[PRIV_X:.*_QFtestEx.*\.omp]] : f32
// CHECK-DAG: omp.private {type = private} @[[PRIV_Y:.*_QFtestEy.*\.omp]] : f32

// CHECK-LABEL: func.func @_QPmulti_local_init
// CHECK:         %[[X:.*]] = fir.alloca f32 <{bindc_name = "x", {{.*}>}}
// CHECK:         %[[X_DECL:.*]]:2 = hlfir.declare %[[X]]
// CHECK:         %[[Y:.*]] = fir.alloca f32 <{bindc_name = "y", {{.*}>}}
// CHECK:         %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]]

// Verify: omp.wsloop with both privatizers
// CHECK:         omp.parallel {
// CHECK:           omp.wsloop
// CHECK-SAME:        private(@[[PRIV_X]] %[[X_DECL]]#0 -> %[[PRIV_X_ARG:.*]], @[[PRIV_Y]] %[[Y_DECL]]#0 -> %[[PRIV_Y_ARG:.*]] : !fir.ref<f32>, !fir.ref<f32>)
// CHECK:             omp.loop_nest

// Verify: both inlined copies appear at top of loop body (before user code)
// CHECK:               %[[LOAD_Y:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<f32>
// CHECK-NEXT:          fir.store %[[LOAD_Y]] to %[[PRIV_Y_ARG]] : !fir.ref<f32>
// CHECK:               %[[LOAD_X:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<f32>
// CHECK-NEXT:          fir.store %[[LOAD_X]] to %[[PRIV_X_ARG]] : !fir.ref<f32>

// Verify: user code follows
// CHECK:               hlfir.declare %[[PRIV_X_ARG]]
// CHECK:               hlfir.declare %[[PRIV_Y_ARG]]
// CHECK:               omp.yield
