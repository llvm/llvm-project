! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Tests REAL lowering
subroutine test_real()
  real(4) :: r4
  real(8) :: r8

  r4 = real(z'40', kind=4)
  r8 = real(z'40', kind=8)

end subroutine

! CHECK-LABEL: func @_QPtest_real() {
! CHECK:  %[[VAL_0:.*]] = fir.alloca f32 {bindc_name = "r4", uniq_name = "_QFtest_realEr4"}
! CHECK:  %[[VAL_1:.*]] = fir.alloca f64 {bindc_name = "r8", uniq_name = "_QFtest_realEr8"}
! CHECK:  %[[CST_0:.*]] = arith.constant 8.968310e-44 : f32
! CHECK:  fir.store %[[CST_0]] to %[[VAL_0]] : !fir.ref<f32>
! CHECK:  %[[CST_1:.*]] = arith.constant 3.162020e-322 : f64
! CHECK:  fir.store %[[CST_1]] to %[[VAL_1]] : !fir.ref<f64>
! CHECK:  return
