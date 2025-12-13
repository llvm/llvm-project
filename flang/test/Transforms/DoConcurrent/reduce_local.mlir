// Tests mapping reductions and local from fir to OpenMP.

// RUN: fir-opt --omp-do-concurrent-conversion="map-to=host" %s | FileCheck %s

fir.declare_reduction @add_reduction_i32 : i32 init {
^bb0(%arg0: i32):
  %c0_i32 = arith.constant 0 : i32
  fir.yield(%c0_i32 : i32)
} combiner {
^bb0(%arg0: i32, %arg1: i32):
  %0 = arith.addi %arg0, %arg1 : i32
  fir.yield(%0 : i32)
}
  fir.local {type = local} @_QFdo_concurrent_reduceEl_private_i32 : i32
  func.func @_QPdo_concurrent_reduce() {
  %3 = fir.alloca i32 {bindc_name = "l", uniq_name = "_QFdo_concurrent_reduceEl"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFdo_concurrent_reduceEl"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5 = fir.alloca i32 {bindc_name = "s", uniq_name = "_QFdo_concurrent_reduceEs"}
  %6:2 = hlfir.declare %5 {uniq_name = "_QFdo_concurrent_reduceEs"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c1 = arith.constant 1 : index
  fir.do_concurrent {
    %9 = fir.alloca i32 {bindc_name = "i"}
    %10:2 = hlfir.declare %9 {uniq_name = "_QFdo_concurrent_reduceEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    fir.do_concurrent.loop (%arg0) = (%c1) to (%c1) step (%c1) local(@_QFdo_concurrent_reduceEl_private_i32 %4#0 -> %arg1 : !fir.ref<i32>) reduce(@add_reduction_i32 #fir.reduce_attr<add> %6#0 -> %arg2 : !fir.ref<i32>) {
      %11 = fir.convert %arg0 : (index) -> i32
      fir.store %11 to %10#0 : !fir.ref<i32>
      %12:2 = hlfir.declare %arg1 {uniq_name = "_QFdo_concurrent_reduceEl"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %13:2 = hlfir.declare %arg2 {uniq_name = "_QFdo_concurrent_reduceEs"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %c1_i32_0 = arith.constant 1 : i32
      hlfir.assign %c1_i32_0 to %12#0 : i32, !fir.ref<i32>
      %14 = fir.load %13#0 : !fir.ref<i32>
      %15 = fir.load %12#0 : !fir.ref<i32>
      %16 = arith.addi %14, %15 : i32
      hlfir.assign %16 to %13#0 : i32, !fir.ref<i32>
    }
  }
  return
}

// CHECK-LABEL:   omp.declare_reduction @add_reduction_i32.omp : i32 init {
// CHECK:         ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           omp.yield(%[[VAL_1]] : i32)

// CHECK-LABEL:   } combiner {
// CHECK:         ^bb0(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32):
// CHECK:           %[[VAL_2:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           omp.yield(%[[VAL_2]] : i32)
// CHECK:         }

// CHECK:         omp.private {type = private} @_QFdo_concurrent_reduceEl_private_i32.omp : i32

// CHECK-LABEL:   func.func @_QPdo_concurrent_reduce() {
// CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i"}
// CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFdo_concurrent_reduceEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "l", uniq_name = "_QFdo_concurrent_reduceEl"}
// CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFdo_concurrent_reduceEl"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "s", uniq_name = "_QFdo_concurrent_reduceEs"}
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFdo_concurrent_reduceEs"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           omp.parallel {
// CHECK:             %[[VAL_7:.*]] = fir.alloca i32 {bindc_name = "i"}
// CHECK:             %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] {uniq_name = "_QFdo_concurrent_reduceEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:             omp.wsloop private(@_QFdo_concurrent_reduceEl_private_i32.omp %[[VAL_3]]#0 -> %[[VAL_9:.*]] : !fir.ref<i32>) reduction(@add_reduction_i32.omp %[[VAL_5]]#0 -> %[[VAL_10:.*]] : !fir.ref<i32>) {
// CHECK:               omp.loop_nest (%[[VAL_11:.*]]) : index = (%[[VAL_6]]) to (%[[VAL_6]]) inclusive step (%[[VAL_6]]) {
// CHECK:                 %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (index) -> i32
// CHECK:                 fir.store %[[VAL_12]] to %[[VAL_8]]#0 : !fir.ref<i32>
// CHECK:                 %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_9]] {uniq_name = "_QFdo_concurrent_reduceEl"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:                 %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_10]] {uniq_name = "_QFdo_concurrent_reduceEs"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:                 %[[VAL_15:.*]] = arith.constant 1 : i32
// CHECK:                 hlfir.assign %[[VAL_15]] to %[[VAL_13]]#0 : i32, !fir.ref<i32>
// CHECK:                 %[[VAL_16:.*]] = fir.load %[[VAL_14]]#0 : !fir.ref<i32>
// CHECK:                 %[[VAL_17:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<i32>
// CHECK:                 %[[VAL_18:.*]] = arith.addi %[[VAL_16]], %[[VAL_17]] : i32
// CHECK:                 hlfir.assign %[[VAL_18]] to %[[VAL_14]]#0 : i32, !fir.ref<i32>
// CHECK:                 omp.yield
// CHECK:               }
// CHECK:             }
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }

