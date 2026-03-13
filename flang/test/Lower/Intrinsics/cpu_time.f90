! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPcpu_time_test(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<f32> {fir.bindc_name = "t"})
subroutine cpu_time_test(t)
    real :: t
    ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
    ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg 1 {uniq_name = "_QFcpu_time_testEt"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
    ! CHECK: %[[result64:.*]] = fir.call @_FortranACpuTime()
    ! CHECK: %[[result32:.*]] = fir.convert %[[result64]] : (f64) -> f32
    ! CHECK: fir.store %[[result32]] to %[[VAL_1]]#0 : !fir.ref<f32>
    call cpu_time(t)
  end subroutine
