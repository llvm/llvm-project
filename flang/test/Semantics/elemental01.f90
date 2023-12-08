! RUN: %python %S/test_errors.py %s %flang_fc1
! Tests ELEMENTAL subprogram constraints C15100-15102

!ERROR: An ELEMENTAL subroutine may not have an alternate return dummy argument
elemental subroutine altret(*)
end subroutine

elemental subroutine arrarg(a)
  !ERROR: A dummy argument of an ELEMENTAL procedure must be scalar
  real, intent(in) :: a(1)
end subroutine

elemental subroutine alloarg(a)
  !ERROR: A dummy argument of an ELEMENTAL procedure may not be ALLOCATABLE
  real, intent(in), allocatable :: a
end subroutine

elemental subroutine coarg(a)
  !ERROR: A dummy argument of an ELEMENTAL procedure may not be a coarray
  real, intent(in) :: a[*]
end subroutine

elemental subroutine ptrarg(a)
  !ERROR: A dummy argument of an ELEMENTAL procedure may not be a POINTER
  real, intent(in), pointer :: a
end subroutine

impure elemental subroutine barearg(a)
  !ERROR: A dummy argument of an ELEMENTAL procedure must have an INTENT() or VALUE attribute
  real :: a
end subroutine

elemental function arrf(n)
  integer, value :: n
  !ERROR: The result of an ELEMENTAL function must be scalar
  real :: arrf(n)
end function

elemental function allof(n)
  integer, value :: n
  !ERROR: The result of an ELEMENTAL function may not be ALLOCATABLE
  real, allocatable :: allof
end function

elemental function ptrf(n)
  integer, value :: n
  !ERROR: The result of an ELEMENTAL function may not be a POINTER
  real, pointer :: ptrf
end function
