// Tests LOCAL + LOCAL_INIT mixed in the same do concurrent loop. Verifies:
//   1. LOCAL variable: omp.private with NO inlined copy
//   2. LOCAL_INIT variable: omp.private with inlined copy at top of loop_nest
//   3. Both privatizers on the same omp.wsloop

// RUN: fir-opt --omp-do-concurrent-conversion="map-to=host" %s | FileCheck %s

fir.local {type = local} @_QFtestEa_private_f32 : f32

fir.local {type = local_init} @_QFtestEb_firstprivate_f32 : f32 copy {
^bb0(%arg0: !fir.ref<f32>, %arg1: !fir.ref<f32>):
  %0 = fir.load %arg0 : !fir.ref<f32>
  fir.store %0 to %arg1 : !fir.ref<f32>
  fir.yield(%arg1 : !fir.ref<f32>)
}

func.func @_QPlocal_and_local_init() {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %0 = fir.alloca f32 {bindc_name = "a", uniq_name = "_QFtestEa"}
  %1:2 = hlfir.declare %0 uniq_name("_QFtestEa") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)

  %2 = fir.alloca f32 {bindc_name = "b", uniq_name = "_QFtestEb"}
  %3:2 = hlfir.declare %2 uniq_name("_QFtestEb") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)

  %cst = arith.constant 5.000000e+00 : f32
  fir.store %cst to %3#0 : !fir.ref<f32>

  fir.do_concurrent {
    %4 = fir.alloca i32 {bindc_name = "i"}
    %5:2 = hlfir.declare %4 uniq_name("_QFtestEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

    fir.do_concurrent.loop (%arg0) = (%c1) to (%c10) step (%c1)
      local(@_QFtestEa_private_f32 %1#0 -> %arg1, @_QFtestEb_firstprivate_f32 %3#0 -> %arg2 : !fir.ref<f32>, !fir.ref<f32>) {
      %6 = fir.convert %arg0 : (index) -> i32
      fir.store %6 to %5#0 : !fir.ref<i32>
      %7:2 = hlfir.declare %arg1 uniq_name("_QFtestEa") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
      %8:2 = hlfir.declare %arg2 uniq_name("_QFtestEb") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
      %9 = fir.load %8#0 : !fir.ref<f32>
      hlfir.assign %9 to %7#0 : f32, !fir.ref<f32>
    }
  }
  return
}

// Verify: both omp.private ops exist (both type=private, no copy_region)
// CHECK-DAG: omp.private {type = private} @[[PRIV_A:.*_QFtestEa.*\.omp]] : f32
// CHECK-DAG: omp.private {type = private} @[[PRIV_B:.*_QFtestEb.*\.omp]] : f32

// CHECK-LABEL: func.func @_QPlocal_and_local_init
// CHECK:         %[[A:.*]] = fir.alloca f32 <{bindc_name = "a", {{.*}>}}
// CHECK:         %[[A_DECL:.*]]:2 = hlfir.declare %[[A]]
// CHECK:         %[[B:.*]] = fir.alloca f32 <{bindc_name = "b", {{.*}>}}
// CHECK:         %[[B_DECL:.*]]:2 = hlfir.declare %[[B]]

// Verify: omp.wsloop has both privatizers
// CHECK:         omp.parallel {
// CHECK:           omp.wsloop
// CHECK-SAME:        private(@[[PRIV_A]] %[[A_DECL]]#0 -> %[[PRIV_A_ARG:.*]], @[[PRIV_B]] %[[B_DECL]]#0 -> %[[PRIV_B_ARG:.*]] : !fir.ref<f32>, !fir.ref<f32>)
// CHECK:             omp.loop_nest

// Verify: LOCAL_INIT (b) has inlined copy at top of loop body
// CHECK:               %[[LOADED_B:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<f32>
// CHECK-NEXT:          fir.store %[[LOADED_B]] to %[[PRIV_B_ARG]] : !fir.ref<f32>

// Verify: user code follows — LOCAL (a) has NO inlined copy
// CHECK:               hlfir.declare %[[PRIV_A_ARG]]
// CHECK:               hlfir.declare %[[PRIV_B_ARG]]
// CHECK:               omp.yield
