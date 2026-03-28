! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine test_dreal(r, c)
  real(8), intent(out) :: r
  complex(8), intent(in) :: c

! CHECK-LABEL: func.func @_QPtest_dreal(
! CHECK-SAME: %[[ARG_0:.*]]: !fir.ref<f64> {fir.bindc_name = "r"},
! CHECK-SAME: %[[ARG_1:.*]]: !fir.ref<complex<f64>> {fir.bindc_name = "c"}) {
! CHECK:   %[[DS:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:   %[[C:.*]]:2 = hlfir.declare %[[ARG_1]] dummy_scope %[[DS]]
! CHECK:   %[[R:.*]]:2 = hlfir.declare %[[ARG_0]] dummy_scope %[[DS]]
! CHECK:   %[[VAL_0:.*]] = fir.load %[[C]]#0 : !fir.ref<complex<f64>>
! CHECK:   %[[VAL_1:.*]] = fir.extract_value %[[VAL_0]], [0 : index] : (complex<f64>) -> f64
! CHECK:   hlfir.assign %[[VAL_1]] to %[[R]]#0 : f64, !fir.ref<f64>
! CHECK:   return
! CHECK: }

  r = dreal(c)
end
