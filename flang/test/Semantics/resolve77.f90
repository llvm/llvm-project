! RUN: %python %S/test_errors.py %s %flang_fc1
! Tests valid and invalid usage of forward references to procedures
! in specification expressions.
module m
  interface ifn2
    module procedure if2
  end interface
  interface ifn3
    module procedure if3
  end interface
  !ERROR: Automatic data object 'a' may not appear in a module
  real :: a(if1(1))
  !ERROR: Automatic data object 'b' may not appear in a module
  real :: b(ifn2(1))
  !ERROR: Automatic data object 'c' may not appear in COMMON block /blk/
  real :: c(if1(1))
  !ERROR: Automatic data object 'd' may not appear in COMMON block //
  real :: d(ifn2(1))
  common /blk/c
  common d
 contains
  subroutine t1(n)
    integer :: iarr(if1(n))
  end subroutine
  pure integer function if1(n)
    integer, intent(in) :: n
    if1 = n
  end function
  subroutine t2(n)
    integer :: iarr(ifn3(n)) ! should resolve to if3
  end subroutine
  pure integer function if2(n)
    integer, intent(in) :: n
    if2 = n
  end function
  pure integer function if3(n)
    integer, intent(in) :: n
    if3 = n
  end function
end module

subroutine nester
  !ERROR: The internal function 'if1' may not be referenced in a specification expression
  real :: a(if1(1))
 contains
  subroutine t1(n)
    !ERROR: The internal function 'if2' may not be referenced in a specification expression
    integer :: iarr(if2(n))
  end subroutine
  pure integer function if1(n)
    integer, intent(in) :: n
    if1 = n
  end function
  pure integer function if2(n)
    integer, intent(in) :: n
    if2 = n
  end function
end subroutine
