!RUN: bbc -emit-hlfir %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine test_rtc(time)
  real(8) :: time
  time = rtc()
end subroutine
! CHECK-LABEL:   func.func @_QPtest_rtc(
! CHECK-SAME:                                %[[VAL_0:.*]]: !fir.ref<f64> {fir.bindc_name = "time"}) {
! CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_2]] arg {{[0-9]+}} {uniq_name = "_QFtest_rtcEtime"} : (!fir.ref<f64>, !fir.dscope) -> (!fir.ref<f64>, !fir.ref<f64>)
! CHECK:           %[[VAL_4:.*]] = fir.call @_FortranAtime() fastmath<contract> : () -> i64
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> f64
! CHECK:           hlfir.assign %[[VAL_5]] to %[[VAL_3]]#0 : f64, !fir.ref<f64>
! CHECK:           return
! CHECK:         }
