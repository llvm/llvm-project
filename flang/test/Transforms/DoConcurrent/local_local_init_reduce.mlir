// Tests LOCAL + LOCAL_INIT + REDUCE all together in the same do concurrent
// loop. Verifies:
//   1. LOCAL: omp.private, no inlined copy
//   2. LOCAL_INIT: omp.private, inlined copy at top of loop_nest
//   3. REDUCE: omp.declare_reduction, reduction clause on wsloop
//   4. All three on the same omp.wsloop

// RUN: fir-opt --omp-do-concurrent-conversion="map-to=host" %s | FileCheck %s

fir.local {type = local} @_QFtestEa_private_f32 : f32

fir.local {type = local_init} @_QFtestEb_firstprivate_f32 : f32 copy {
^bb0(%arg0: !fir.ref<f32>, %arg1: !fir.ref<f32>):
  %0 = fir.load %arg0 : !fir.ref<f32>
  fir.store %0 to %arg1 : !fir.ref<f32>
  fir.yield(%arg1 : !fir.ref<f32>)
}

fir.declare_reduction @add_reduction_f32 : f32 init {
^bb0(%arg0: f32):
  %cst = arith.constant 0.000000e+00 : f32
  fir.yield(%cst : f32)
} combiner {
^bb0(%arg0: f32, %arg1: f32):
  %0 = arith.addf %arg0, %arg1 fastmath<contract> : f32
  fir.yield(%0 : f32)
}

func.func @_QPlocal_local_init_reduce() {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %0 = fir.alloca f32 {bindc_name = "a", uniq_name = "_QFtestEa"}
  %1:2 = hlfir.declare %0 uniq_name("_QFtestEa") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)

  %2 = fir.alloca f32 {bindc_name = "b", uniq_name = "_QFtestEb"}
  %3:2 = hlfir.declare %2 uniq_name("_QFtestEb") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)

  %4 = fir.alloca f32 {bindc_name = "sum", uniq_name = "_QFtestEsum"}
  %5:2 = hlfir.declare %4 uniq_name("_QFtestEsum") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)

  %cst = arith.constant 5.000000e+00 : f32
  fir.store %cst to %3#0 : !fir.ref<f32>

  fir.do_concurrent {
    %6 = fir.alloca i32 {bindc_name = "i"}
    %7:2 = hlfir.declare %6 uniq_name("_QFtestEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

    fir.do_concurrent.loop (%arg0) = (%c1) to (%c10) step (%c1)
      local(@_QFtestEa_private_f32 %1#0 -> %arg1, @_QFtestEb_firstprivate_f32 %3#0 -> %arg2 : !fir.ref<f32>, !fir.ref<f32>)
      reduce(@add_reduction_f32 #fir.reduce_attr<add> %5#0 -> %arg3 : !fir.ref<f32>) {
      %8 = fir.convert %arg0 : (index) -> i32
      fir.store %8 to %7#0 : !fir.ref<i32>
      %9:2 = hlfir.declare %arg1 uniq_name("_QFtestEa") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
      %10:2 = hlfir.declare %arg2 uniq_name("_QFtestEb") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
      %11:2 = hlfir.declare %arg3 uniq_name("_QFtestEsum") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
      %12 = fir.load %10#0 : !fir.ref<f32>
      hlfir.assign %12 to %9#0 : f32, !fir.ref<f32>
      %13 = fir.load %9#0 : !fir.ref<f32>
      %14 = fir.load %11#0 : !fir.ref<f32>
      %15 = arith.addf %14, %13 fastmath<contract> : f32
      hlfir.assign %15 to %11#0 : f32, !fir.ref<f32>
    }
  }
  return
}

// Verify: omp.private ops for both LOCAL and LOCAL_INIT (both type=private)
// CHECK-DAG: omp.private {type = private} @[[PRIV_A:.*_QFtestEa.*\.omp]] : f32
// CHECK-DAG: omp.private {type = private} @[[PRIV_B:.*_QFtestEb.*\.omp]] : f32

// Verify: omp.declare_reduction for the REDUCE var
// CHECK:      omp.declare_reduction @[[REDUCER:.*add_reduction_f32\.omp]] : f32 init {
// CHECK:         %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         omp.yield(%[[CST]] : f32)
// CHECK:       } combiner {
// CHECK:         %[[ADD:.*]] = arith.addf %{{.*}}, %{{.*}} fastmath<contract> : f32
// CHECK:         omp.yield(%[[ADD]] : f32)
// CHECK:       }

// CHECK-LABEL: func.func @_QPlocal_local_init_reduce
// CHECK:         %[[A:.*]] = fir.alloca f32 <{bindc_name = "a", {{.*}>}}
// CHECK:         %[[A_DECL:.*]]:2 = hlfir.declare %[[A]]
// CHECK:         %[[B:.*]] = fir.alloca f32 <{bindc_name = "b", {{.*}>}}
// CHECK:         %[[B_DECL:.*]]:2 = hlfir.declare %[[B]]
// CHECK:         %[[SUM:.*]] = fir.alloca f32 <{bindc_name = "sum", {{.*}>}}
// CHECK:         %[[SUM_DECL:.*]]:2 = hlfir.declare %[[SUM]]

// Verify: omp.wsloop has private (LOCAL + LOCAL_INIT) and reduction clauses
// CHECK:         omp.parallel {
// CHECK:           omp.wsloop
// CHECK-SAME:        private(@[[PRIV_A]] %[[A_DECL]]#0 -> %[[PRIV_A_ARG:.*]], @[[PRIV_B]] %[[B_DECL]]#0 -> %[[PRIV_B_ARG:.*]] : !fir.ref<f32>, !fir.ref<f32>)
// CHECK-SAME:        reduction(@[[REDUCER]] %[[SUM_DECL]]#0 -> %[[RED_ARG:.*]] : !fir.ref<f32>)
// CHECK:             omp.loop_nest

// Verify: LOCAL_INIT (b) has inlined copy at top — LOCAL (a) does NOT
// CHECK:               %[[LOADED_B:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<f32>
// CHECK-NEXT:          fir.store %[[LOADED_B]] to %[[PRIV_B_ARG]] : !fir.ref<f32>

// Verify: user code follows
// CHECK:               hlfir.declare %[[PRIV_A_ARG]]
// CHECK:               hlfir.declare %[[PRIV_B_ARG]]
// CHECK:               hlfir.declare %[[RED_ARG]]
// CHECK:               omp.yield
