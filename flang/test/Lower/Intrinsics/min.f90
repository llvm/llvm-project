! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s
! Test that min/max A(X>2) optional arguments are handled regardless
! of the order in which they appear. Only A1 and A2 are mandatory.

real function test(a, b, c)
  real, optional :: a, b, c
  test = min(a1=a, a3=c, a2=c)
end function
! CHECK-LABEL:   func.func @_QPtest(
! CHECK-SAME:                       %[[VAL_0:.*]]: !fir.ref<f32> {fir.bindc_name = "a", fir.optional},
! CHECK-SAME:                       %[[VAL_1:.*]]: !fir.ref<f32> {fir.bindc_name = "b", fir.optional},
! CHECK-SAME:                       %[[VAL_2:.*]]: !fir.ref<f32> {fir.bindc_name = "c", fir.optional}) -> f32 {
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}}uniq_name = "_QFtestEa"}
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] {{.*}}uniq_name = "_QFtestEb"}
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_2]] {{.*}}uniq_name = "_QFtestEc"}
! CHECK:           %[[VAL_6:.*]] = fir.alloca f32 {bindc_name = "test", uniq_name = "_QFtestEtest"}
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFtestEtest"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<f32>
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<f32>
! CHECK:           %[[VAL_10:.*]] = fir.is_present %[[VAL_5]]#0 : (!fir.ref<f32>) -> i1
! CHECK:           %[[VAL_11:.*]] = arith.cmpf olt, %[[VAL_8]], %[[VAL_9]] : f32
! CHECK:           %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_8]], %[[VAL_9]] : f32
! CHECK:           %[[VAL_13:.*]] = fir.if %[[VAL_10]] -> (f32) {
! CHECK:             %[[VAL_14:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<f32>
! CHECK:             %[[VAL_15:.*]] = arith.cmpf olt, %[[VAL_12]], %[[VAL_14]] : f32
! CHECK:             %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_12]], %[[VAL_14]] : f32
! CHECK:             fir.result %[[VAL_16]] : f32
! CHECK:           } else {
! CHECK:             fir.result %[[VAL_12]] : f32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_13]] to %[[VAL_7]]#0 : f32, !fir.ref<f32>
! CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_7]]#1 : !fir.ref<f32>
! CHECK:           return %[[VAL_17]] : f32
! CHECK:         }
