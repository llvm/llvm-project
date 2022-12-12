! Test lowering of unary intrinsic operations to HLFIR
! RUN: bbc -emit-fir -hlfir -o - %s 2>&1 | FileCheck %s

subroutine test_not(l, x)
  logical :: l, x
  l = .not.x
end subroutine
! CHECK-LABEL: func.func @_QPtest_not(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}}x"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:  %[[VAL_5:.*]] = arith.constant true
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_4]] : (!fir.logical<4>) -> i1
! CHECK:  %[[VAL_7:.*]] = arith.xori %[[VAL_6]], %[[VAL_5]] : i1

subroutine test_negate_int(res, x)
  integer :: res, x
  res = -x
end subroutine
! CHECK-LABEL: func.func @_QPtest_negate_int(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}}x"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_6:.*]] = arith.subi %[[VAL_5]], %[[VAL_4]] : i32

subroutine test_negate_real(res, x)
  real :: res, x
  res = -x
end subroutine
! CHECK-LABEL: func.func @_QPtest_negate_real(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}}x"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_5:.*]] = arith.negf %[[VAL_4]] fastmath<contract> : f32

subroutine test_negate_complex(res, x)
  complex :: res, x
  res = -x
end subroutine
! CHECK-LABEL: func.func @_QPtest_negate_complex(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}}x"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_5:.*]] = fir.negc %[[VAL_4]] : !fir.complex<4>

subroutine test_complex_component_real(res, x)
  real :: res
  complex :: x
  res = real(x)
end subroutine
! CHECK-LABEL: func.func @_QPtest_complex_component_real(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}}x"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_5:.*]] = fir.extract_value %[[VAL_4]], [0 : index] : (!fir.complex<4>) -> f32

subroutine test_complex_component_imag(res, x)
  real :: res
  complex :: x
  res = aimag(x)
end subroutine
! CHECK-LABEL: func.func @_QPtest_complex_component_imag(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}}x"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_5:.*]] = fir.extract_value %[[VAL_4]], [1 : index] : (!fir.complex<4>) -> f32
