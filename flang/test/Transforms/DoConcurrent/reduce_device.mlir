// RUN: fir-opt --omp-do-concurrent-conversion="map-to=device" %s -o - | FileCheck %s

fir.declare_reduction @add_reduction_f32 : f32 init {
^bb0(%arg0: f32):
  %cst = arith.constant 0.000000e+00 : f32
  fir.yield(%cst : f32)
} combiner {
^bb0(%arg0: f32, %arg1: f32):
  %0 = arith.addf %arg0, %arg1 fastmath<contract> : f32
  fir.yield(%0 : f32)
}

func.func @_QPfoo() {
  %0 = fir.dummy_scope : !fir.dscope
  %3 = fir.alloca f32 {bindc_name = "s", uniq_name = "_QFfooEs"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFfooEs"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
    %c1 = arith.constant 1 : index
  %c10 = arith.constant 1 : index
  fir.do_concurrent {
    %7 = fir.alloca i32 {bindc_name = "i"}
    %8:2 = hlfir.declare %7 {uniq_name = "_QFfooEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    fir.do_concurrent.loop (%arg0) = (%c1) to (%c10) step (%c1) reduce(@add_reduction_f32 #fir.reduce_attr<add> %4#0 -> %arg1 : !fir.ref<f32>) {
      %9 = fir.convert %arg0 : (index) -> i32
      fir.store %9 to %8#0 : !fir.ref<i32>
      %10:2 = hlfir.declare %arg1 {uniq_name = "_QFfooEs"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
      %11 = fir.load %10#0 : !fir.ref<f32>
      %cst = arith.constant 1.000000e+00 : f32
      %12 = arith.addf %11, %cst fastmath<contract> : f32
      hlfir.assign %12 to %10#0 : f32, !fir.ref<f32>
    }
  }
  return
}

// CHECK: omp.declare_reduction @[[OMP_RED:.*.omp]] : f32

// CHECK: %[[S_DECL:.*]]:2 = hlfir.declare %6 {uniq_name = "_QFfooEs"}
// CHECK: %[[S_MAP:.*]] = omp.map.info var_ptr(%[[S_DECL]]#1

// CHECK: omp.target host_eval({{.*}}) map_entries({{.*}}, %[[S_MAP]] -> %[[S_TARGET_ARG:.*]] : {{.*}}) {
// CHECK:   %[[S_DEV_DECL:.*]]:2 = hlfir.declare %[[S_TARGET_ARG]]
// CHECK:   omp.teams reduction(@[[OMP_RED]] %[[S_DEV_DECL]]#0 -> %[[RED_TEAMS_ARG:.*]] : !fir.ref<f32>) {
// CHECK:   omp.parallel {
// CHECK:     omp.distribute {
// CHECK:       omp.wsloop reduction(@[[OMP_RED]] %[[RED_TEAMS_ARG]] -> %[[RED_WS_ARG:.*]] : {{.*}}) {
// CHECK:         %[[S_WS_DECL:.*]]:2 = hlfir.declare %[[RED_WS_ARG]] {uniq_name = "_QFfooEs"}
// CHECK:         %[[S_VAL:.*]] = fir.load %[[S_WS_DECL]]#0
// CHECK:         %[[RED_RES:.*]] = arith.addf %[[S_VAL]], %{{.*}} fastmath<contract> : f32
// CHECK:         hlfir.assign %[[RED_RES]] to %[[S_WS_DECL]]#0
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: }
