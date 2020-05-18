! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPchar_return_callee(%arg0: !fir.ref<!fir.char<1>>, %arg1: index, %arg2: !fir.ref<i32>) -> !fir.boxchar<1>
function char_return_callee(i)
  character(10) :: char_return_callee
  integer :: i
end function

! FIXME: the mangling is incorrect.
! CHECK-LABEL: func @_QFtest_char_return_callerPchar_return_caller(!fir.ref<!fir.char<1>>, index, !fir.ref<i32>) -> !fir.boxchar<1>
subroutine test_char_return_caller
  character(10) :: char_return_caller
  print *, char_return_caller(5)
end subroutine

! TODO more implicit interface cases with/without explicit interface

