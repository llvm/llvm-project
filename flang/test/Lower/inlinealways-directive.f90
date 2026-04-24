! Check the appropriate flags are added to inline functions when inlinealways is used, or ignored when the name is incorrect.

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine test_subroutine()
  !DIR$ INLINEALWAYS test_subroutine
end subroutine
! CHECK: func.func @_QPtest_subroutine() attributes {llvm.always_inline} {

subroutine test_subroutine2()
  !DIR$ INLINEALWAYS wrong_subroutine
end subroutine
! CHECK: func.func @_QPtest_subroutine2() {

subroutine test_subroutine3()
end subroutine

function test_func1()
!DIR$ INLINEALWAYS test_func1
end function
! CHECK: func.func @_QPtest_func1() -> f32 attributes {llvm.always_inline} {

function test_func2()
!DIR$ INLINEALWAYS wrong_func
end function
! CHCEK: func.func @_QPtest_func2() -> f32 {

integer function test_func3() result(res)
  res = 10
end function

subroutine test()
  implicit none
  integer:: result, test_func3

  !DIR$ INLINEALWAYS
  call test_subroutine3
! CHECK: fir.call @_QPtest_subroutine3() fastmath<contract> {inline_attr = #fir.inline_attrs<always_inline>} : () -> ()

  result = 0
  !DIR$ INLINEALWAYS
  result = test_func3()
! CHCEK: %[[.*]] = fir.call @_QPtest_func3() fastmath<contract> {inline_attr = #fir.inline_attrs<always_inline>} : () -> i32
end subroutine
