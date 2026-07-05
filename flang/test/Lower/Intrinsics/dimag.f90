! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine test_dimag(r, c)
  real(8), intent(out) :: r
  complex(8), intent(in) :: c

! CHECK-LABEL: func @_QPtest_dimag(
! CHECK-SAME: %[[ARG_0:.*]]: !fir.ref<f64> {fir.bindc_name = "r"},
! CHECK-SAME: %[[ARG_1:.*]]: !fir.ref<complex<f64>> {fir.bindc_name = "c"}) {
! CHECK:   %[[DS:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:   %[[C_DECL:.*]]:2 = hlfir.declare %[[ARG_1]] dummy_scope %[[DS]] {{.*}}
! CHECK:   %[[R_DECL:.*]]:2 = hlfir.declare %[[ARG_0]] dummy_scope %[[DS]] {{.*}}
! CHECK:   %[[VAL_0:.*]] = fir.load %[[C_DECL]]#0 : !fir.ref<complex<f64>>
! CHECK:   %[[VAL_1:.*]] = fir.extract_value %[[VAL_0]], [1 : index] : (complex<f64>) -> f64
! CHECK:   hlfir.assign %[[VAL_1]] to %[[R_DECL]]#0 : f64, !fir.ref<f64>
! CHECK:   return
! CHECK: }

  r = dimag(c)
end
