! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Check OpenACC clause validity for the following construct and directive:
!   2.15.1 routine

module openacc_routine_validity
  implicit none

  !$acc routine(sub3) seq

  !$acc routine(fct2) vector

  !$acc routine(sub3)

  !ERROR: ROUTINE directive without name must appear within the specification part of a subroutine or function definition, or within an interface body for a subroutine or function in an interface block
  !$acc routine seq

  !$acc routine(dummy) seq

contains

  subroutine sub1(a)
    real :: a(:)
    !$acc routine
  end subroutine sub1

  subroutine sub2(a)
    real :: a(:)
    !ERROR: Clause NOHOST is not allowed after clause DEVICE_TYPE on the ROUTINE directive
    !$acc routine seq device_type(*) nohost
  end subroutine sub2

  subroutine sub3(a)
    real :: a(:)
  end subroutine sub3

  subroutine sub4(a)
    real :: a(:)
    !$acc routine seq
  end subroutine sub4

  subroutine sub5(a)
    real :: a(:)
    !$acc routine(sub5) seq
  end subroutine sub5

  function fct1(a)
    integer :: fct1
    real :: a(:)
    !$acc routine vector nohost
  end function fct1

  function fct2(a)
    integer :: fct2
    real :: a(:)
  end function fct2

  function fct3(a)
    integer :: fct3
    real :: a(:)
    !$acc routine seq bind(fct2)
  end function fct3

  function fct4(a)
    integer :: fct4
    real :: a(:)
    !$acc routine seq bind("_fct4")
  end function fct4

  subroutine sub6(a)
    real :: a(:)
    !$acc routine seq bind(dummy_sub)
  end subroutine sub6

end module openacc_routine_validity
