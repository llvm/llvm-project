! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine test_dconjg(r, c)
  complex(8), intent(out) :: r
  complex(8), intent(in) :: c

! CHECK-LABEL: func @_QPtest_dconjg(
! CHECK-SAME: %[[ARG_0:.*]]: !fir.ref<complex<f64>> {fir.bindc_name = "r"},
! CHECK-SAME: %[[ARG_1:.*]]: !fir.ref<complex<f64>> {fir.bindc_name = "c"}) {
! CHECK:   %[[DS:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:   %[[C_DECL:.*]]:2 = hlfir.declare %[[ARG_1]] dummy_scope %[[DS]] {{.*}}
! CHECK:   %[[R_DECL:.*]]:2 = hlfir.declare %[[ARG_0]] dummy_scope %[[DS]] {{.*}}
! CHECK:   %[[VAL_0:.*]] = fir.load %[[C_DECL]]#0 : !fir.ref<complex<f64>>
! CHECK:   %[[VAL_1:.*]] = fir.extract_value %[[VAL_0]], [1 : index] : (complex<f64>) -> f64
! CHECK:   %[[VAL_2:.*]] = arith.negf %[[VAL_1]] {{.*}}: f64
! CHECK:   %[[VAL_3:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_2]], [1 : index] : (complex<f64>, f64) -> complex<f64>
! CHECK:   hlfir.assign %[[VAL_3]] to %[[R_DECL]]#0 : complex<f64>, !fir.ref<complex<f64>>
! CHECK:   return
! CHECK: }

  r = dconjg(c)
end
