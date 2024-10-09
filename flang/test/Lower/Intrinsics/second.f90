!RUN: bbc -emit-hlfir %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine test_subroutine(time)
  real :: time
  call second(time)
end subroutine
! CHECK-LABEL:   func.func @_QPtest_subroutine(
! CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.ref<f32> {fir.bindc_name = "time"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFtest_subroutineEtime"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[VAL_3:.*]] = fir.call @_FortranACpuTime() fastmath<contract> : () -> f64
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (f64) -> f32
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_2]]#1 : !fir.ref<f32>
! CHECK:           return
! CHECK:         }


subroutine test_function(time)
  real :: time
  time = second()
end subroutine
! CHECK-LABEL:   func.func @_QPtest_function(
! CHECK-SAME:                                %[[VAL_0:.*]]: !fir.ref<f32> {fir.bindc_name = "time"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca f32
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] {uniq_name = "_QFtest_functionEtime"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[VAL_4:.*]] = fir.call @_FortranACpuTime() fastmath<contract> : () -> f64
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (f64) -> f32
! CHECK:           fir.store %[[VAL_5]] to %[[VAL_1]] : !fir.ref<f32>
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_1]] : !fir.ref<f32>
! CHECK:           hlfir.assign %[[VAL_6]] to %[[VAL_3]]#0 : f32, !fir.ref<f32>
! CHECK:           return
! CHECK:         }

subroutine test_function_subexpr(t1, t2)
  real :: t1, t2
  t2 = second() - t1
end subroutine
! CHECK-LABEL:   func.func @_QPtest_function_subexpr(
! CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.ref<f32> {fir.bindc_name = "t1"},
! CHECK-SAME:                                        %[[VAL_1:.*]]: !fir.ref<f32> {fir.bindc_name = "t2"}) {
! CHECK:           %[[VAL_2:.*]] = fir.alloca f32
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_3]] {uniq_name = "_QFtest_function_subexprEt1"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_3]] {uniq_name = "_QFtest_function_subexprEt2"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[VAL_6:.*]] = fir.call @_FortranACpuTime() fastmath<contract> : () -> f64
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (f64) -> f32
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_2]] : !fir.ref<f32>
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_2]] : !fir.ref<f32>
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<f32>
! CHECK:           %[[VAL_10:.*]] = arith.subf %[[VAL_8]], %[[VAL_9]] fastmath<contract> : f32
! CHECK:           hlfir.assign %[[VAL_10]] to %[[VAL_5]]#0 : f32, !fir.ref<f32>
! CHECK:           return
! CHECK:         }
