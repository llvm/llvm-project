!RUN: bbc -emit-hlfir %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine test_timef(t)
  real(8) :: t
  t = timef()
end subroutine
! CHECK-LABEL:   func.func @_QPtest_timef(
! CHECK-SAME:      %[[VAL_0:.*]]: !fir.ref<f64> {fir.bindc_name = "t"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QFtest_timefEt"} : (!fir.ref<f64>, !fir.dscope) -> (!fir.ref<f64>, !fir.ref<f64>)
! CHECK:           %[[VAL_4:.*]] = fir.call @_FortranATimef() fastmath<contract> : () -> f64
! CHECK:           hlfir.assign %[[VAL_4]] to %[[VAL_3]]#0 : f64, !fir.ref<f64>
! CHECK:           return
! CHECK:         }
