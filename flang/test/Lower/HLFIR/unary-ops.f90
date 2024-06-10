! Test lowering of unary intrinsic operations to HLFIR
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

subroutine parentheses_numeric_var(x)
  integer :: x
  call bar((x))
end subroutine
! CHECK-LABEL: func.func @_QPparentheses_numeric_var(
! CHECK:  %[[VAL_2:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK:  %[[VAL_3:.*]] = hlfir.no_reassoc %[[VAL_2]] : i32

subroutine parentheses_numeric_expr(x)
  real :: x
  call bar((x+1000.)*2.)
end subroutine
! CHECK-LABEL: func.func @_QPparentheses_numeric_expr(
! CHECK:  %[[VAL_4:.*]] = arith.addf
! CHECK:  %[[VAL_5:.*]] = hlfir.no_reassoc %[[VAL_4]] : f32
! CHECK:  %[[VAL_7:.*]] = arith.mulf %[[VAL_5]], %{{.*}}

subroutine parentheses_char_var(x)
  character(*) :: x
  call bar2((x))
end subroutine
! CHECK-LABEL: func.func @_QPparentheses_char_var(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare
! CHECK:  %[[VAL_3:.*]] = hlfir.as_expr %[[VAL_2]]#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>

subroutine parentheses_char_expr(x)
  character(*) :: x
  call bar2((x//x)//x)
end subroutine
! CHECK-LABEL: func.func @_QPparentheses_char_expr(
! CHECK:  %[[VAL_4:.*]] = hlfir.concat
! CHECK:  %[[VAL_5:.*]] = hlfir.no_reassoc %[[VAL_4]] : !hlfir.expr<!fir.char<1,?>>
! CHECK:  %[[VAL_7:.*]] = hlfir.concat %[[VAL_5]], %{{.*}} len %{{.*}}
subroutine test_not(l, x)
  logical :: l, x
  l = .not.x
end subroutine
! CHECK-LABEL: func.func @_QPtest_not(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}}x"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:  %[[VAL_5:.*]] = arith.constant true
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_4]] : (!fir.logical<4>) -> i1
! CHECK:  %[[VAL_7:.*]] = arith.xori %[[VAL_6]], %[[VAL_5]] : i1

subroutine test_negate_int(res, x)
  integer :: res, x
  res = -x
end subroutine
! CHECK-LABEL: func.func @_QPtest_negate_int(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}}x"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_6:.*]] = arith.subi %[[VAL_5]], %[[VAL_4]] : i32

subroutine test_negate_real(res, x)
  real :: res, x
  res = -x
end subroutine
! CHECK-LABEL: func.func @_QPtest_negate_real(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}}x"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_5:.*]] = arith.negf %[[VAL_4]] fastmath<contract> : f32

subroutine test_negate_complex(res, x)
  complex :: res, x
  res = -x
end subroutine
! CHECK-LABEL: func.func @_QPtest_negate_complex(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}}x"} : (!fir.ref<!fir.complex<4>>, !fir.dscope) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_5:.*]] = fir.negc %[[VAL_4]] : !fir.complex<4>

subroutine test_complex_component_real(res, x)
  real :: res
  complex :: x
  res = real(x)
end subroutine
! CHECK-LABEL: func.func @_QPtest_complex_component_real(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}}x"} : (!fir.ref<!fir.complex<4>>, !fir.dscope) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_5:.*]] = fir.extract_value %[[VAL_4]], [0 : index] : (!fir.complex<4>) -> f32

subroutine test_complex_component_imag(res, x)
  real :: res
  complex :: x
  res = aimag(x)
end subroutine
! CHECK-LABEL: func.func @_QPtest_complex_component_imag(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}}x"} : (!fir.ref<!fir.complex<4>>, !fir.dscope) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_5:.*]] = fir.extract_value %[[VAL_4]], [1 : index] : (!fir.complex<4>) -> f32
