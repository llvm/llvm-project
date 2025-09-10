// Tests mapping reductions from fir to OpenMP (all regions).

// RUN: fir-opt --omp-do-concurrent-conversion="map-to=host" %s | FileCheck %s

fir.declare_reduction @add_reduction_i32 : i32 init {
^bb0(%arg0: i32):
  fir.yield(%arg0 : i32)
} combiner {
^bb0(%arg0: i32, %arg1: i32):
  fir.yield(%arg0 : i32)
} atomic {
^bb0(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>):
  fir.yield(%arg0 : !fir.ref<i32>)
} cleanup {
^bb0(%arg0: i32):
  fir.yield
}

func.func @_QPdo_concurrent_reduce() {
  %3 = fir.alloca i32 {bindc_name = "s", uniq_name = "_QFdo_concurrent_reduceEs"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFdo_concurrent_reduceEs"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c1 = arith.constant 1 : index
  fir.do_concurrent {
    %7 = fir.alloca i32 {bindc_name = "i"}
    %8:2 = hlfir.declare %7 {uniq_name = "_QFdo_concurrent_reduceEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    fir.do_concurrent.loop (%arg0) = (%c1) to (%c1) step (%c1) reduce(@add_reduction_i32 #fir.reduce_attr<add> %4#0 -> %arg1 : !fir.ref<i32>) {
      %9 = fir.convert %arg0 : (index) -> i32
      fir.store %9 to %8#0 : !fir.ref<i32>
    }
  }
  return
}

// CHECK-LABEL:   omp.declare_reduction @add_reduction_i32.omp : i32 init {
// CHECK:         ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:           omp.yield(%[[VAL_0]] : i32)

// CHECK-LABEL:   } combiner {
// CHECK:         ^bb0(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32):
// CHECK:           omp.yield(%[[VAL_0]] : i32)

// CHECK-LABEL:   } atomic {
// CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>, %[[VAL_1:.*]]: !fir.ref<i32>):
// CHECK:           omp.yield(%[[VAL_0]] : !fir.ref<i32>)

// CHECK-LABEL:   } cleanup {
// CHECK:         ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:           omp.yield
// CHECK:         }

// CHECK-LABEL:   func.func @_QPdo_concurrent_reduce() {
// CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i"}
// CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFdo_concurrent_reduceEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "s", uniq_name = "_QFdo_concurrent_reduceEs"}
// CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFdo_concurrent_reduceEs"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           omp.parallel {
// CHECK:             %[[VAL_5:.*]] = fir.alloca i32 {bindc_name = "i"}
// CHECK:             %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFdo_concurrent_reduceEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:             omp.wsloop reduction(@add_reduction_i32.omp %[[VAL_3]]#0 -> %[[VAL_7:.*]] : !fir.ref<i32>) {
// CHECK:               omp.loop_nest (%[[VAL_8:.*]]) : index = (%[[VAL_4]]) to (%[[VAL_4]]) inclusive step (%[[VAL_4]]) {
// CHECK:                 %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (index) -> i32
// CHECK:                 fir.store %[[VAL_9]] to %[[VAL_6]]#0 : !fir.ref<i32>
// CHECK:                 omp.yield
// CHECK:               }
// CHECK:             }
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
