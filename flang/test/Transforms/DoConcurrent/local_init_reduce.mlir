// Tests LOCAL_INIT + REDUCE in the same do concurrent loop. Verifies:
//   1. omp.private has type=private (no copy_region) for LOCAL_INIT var
//   2. omp.declare_reduction exists for the REDUCE var
//   3. omp.wsloop has both private and reduction clauses
//   4. Per-iteration copy is inlined at top of omp.loop_nest body

// RUN: fir-opt --omp-do-concurrent-conversion="map-to=host" %s | FileCheck %s

fir.local {type = local_init} @_QFtestEtmp_firstprivate_f32 : f32 copy {
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

func.func @_QPlocal_init_reduce() {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %0 = fir.alloca f32 {bindc_name = "tmp", uniq_name = "_QFtestEtmp"}
  %1:2 = hlfir.declare %0 uniq_name("_QFtestEtmp") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)

  %2 = fir.alloca f32 {bindc_name = "sum", uniq_name = "_QFtestEsum"}
  %3:2 = hlfir.declare %2 uniq_name("_QFtestEsum") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)

  %cst = arith.constant 1.000000e+00 : f32
  fir.store %cst to %1#0 : !fir.ref<f32>

  fir.do_concurrent {
    %4 = fir.alloca i32 {bindc_name = "i"}
    %5:2 = hlfir.declare %4 uniq_name("_QFtestEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

    fir.do_concurrent.loop (%arg0) = (%c1) to (%c10) step (%c1)
      local(@_QFtestEtmp_firstprivate_f32 %1#0 -> %arg1 : !fir.ref<f32>)
      reduce(@add_reduction_f32 #fir.reduce_attr<add> %3#0 -> %arg2 : !fir.ref<f32>) {
      %6 = fir.convert %arg0 : (index) -> i32
      fir.store %6 to %5#0 : !fir.ref<i32>
      %7:2 = hlfir.declare %arg1 uniq_name("_QFtestEtmp") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
      %8:2 = hlfir.declare %arg2 uniq_name("_QFtestEsum") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
      %9 = fir.load %7#0 : !fir.ref<f32>
      %10 = fir.load %8#0 : !fir.ref<f32>
      %11 = arith.addf %10, %9 fastmath<contract> : f32
      hlfir.assign %11 to %8#0 : f32, !fir.ref<f32>
    }
  }
  return
}

// Verify: omp.private has type=private with NO copy region for LOCAL_INIT
// CHECK:      omp.private {type = private} @[[PRIVATIZER:.*_QFtestEtmp.*\.omp]] : f32

// Verify: omp.declare_reduction exists for the REDUCE var
// CHECK:      omp.declare_reduction @[[REDUCER:.*add_reduction_f32\.omp]] : f32 init {
// CHECK:         %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         omp.yield(%[[CST]] : f32)
// CHECK:       } combiner {
// CHECK:         %[[ADD:.*]] = arith.addf %{{.*}}, %{{.*}} fastmath<contract> : f32
// CHECK:         omp.yield(%[[ADD]] : f32)
// CHECK:       }

// CHECK-LABEL: func.func @_QPlocal_init_reduce
// CHECK:         %[[TMP:.*]] = fir.alloca f32 <{bindc_name = "tmp", {{.*}>}}
// CHECK:         %[[TMP_DECL:.*]]:2 = hlfir.declare %[[TMP]]
// CHECK:         %[[SUM:.*]] = fir.alloca f32 <{bindc_name = "sum", {{.*}>}}
// CHECK:         %[[SUM_DECL:.*]]:2 = hlfir.declare %[[SUM]]

// Verify: omp.wsloop has both private and reduction clauses
// CHECK:         omp.parallel {
// CHECK:           omp.wsloop private(@[[PRIVATIZER]] %[[TMP_DECL]]#0 -> %[[PRIV:.*]] : !fir.ref<f32>) reduction(@[[REDUCER]] %[[SUM_DECL]]#0 -> %[[RED:.*]] : !fir.ref<f32>)
// CHECK:             omp.loop_nest

// Verify: per-iteration copy inlined at top of loop body (BEFORE user code)
// CHECK:               %[[LOADED:.*]] = fir.load %[[TMP_DECL]]#0 : !fir.ref<f32>
// CHECK-NEXT:          fir.store %[[LOADED]] to %[[PRIV]] : !fir.ref<f32>

// Verify: user code follows the copy
// CHECK:               hlfir.declare %[[PRIV]]
// CHECK:               hlfir.declare %[[RED]]
// CHECK:               omp.yield
