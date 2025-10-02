! RUN: %check_flang_tidy %s performance-pure-procedure %t
module pure_test
  implicit none
contains
  ! This function could be pure but isn't
  ! CHECK-MESSAGES: :[[@LINE+1]]:12: warning: Procedure 'add' could be PURE but is not
  function add(a, b) result(c)
    integer, intent(in) :: a, b
    integer :: c
    c = a + b
  end function add

  ! I/O statements make this procedure impossible to be pure
  subroutine print_sum(a, b)
    integer, intent(in) :: a, b
    print *, "Sum:", a + b
  end subroutine print_sum

  ! A properly declared pure function
  pure function multiply(a, b) result(c)
    integer, intent(in) :: a, b
    integer :: c
    c = a * b
  end function multiply
end module pure_test
