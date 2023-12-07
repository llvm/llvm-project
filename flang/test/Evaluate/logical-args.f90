! Test that actual logical arguments convert to the right kind when it is non-default
! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fdebug-unparse -fdefault-integer-8 %s 2>&1 | FileCheck %s --check-prefixes CHECK-8

module m
  interface foog
     subroutine foo4(l)
       logical(kind=4), intent(in) :: l
     end subroutine foo4

     subroutine foo8(l)
       logical(kind=8), intent(in) :: l
     end subroutine foo8
  end interface foog
end module m

program main
  use m
  integer :: x(10), y
  ! CHECK: CALL foo(.true._4)
  ! CHECK-8: CALL foo(logical(.true._4,kind=8))
  call foo(1 < 2)
  ! CHECK: CALL fooa(x>y)
  ! CHECK-8: CALL fooa(logical(x>y,kind=8))
  call fooa(x > y)

  ! Ensure that calls to interfaces call the correct subroutine
  ! CHECK: CALL foo4(.true._4)
  ! CHECK-8: CALL foo8(.true._8)
  call foog(.true.)

  contains
    subroutine foo(l)
      logical :: l
    end subroutine foo

    subroutine fooa(l)
      logical :: l(10)
    end subroutine fooa

end program main
