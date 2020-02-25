! Tests -fget-definition returning source position of symbol definition.

module m1
 private :: f
contains
 pure subroutine s (x, yyy) bind(c)
  intent(in) :: x
  intent(inout) :: yyy
 contains
  pure subroutine ss
  end subroutine
 end subroutine
 recursive pure function f() result(x)
  real, allocatable :: x
  x = 1.0
 end function
end module

! RUN: echo %t 1>&2;
! RUN: ${F18} -fget-definition 7 17 18 -fparse-only %s > %t;
! RUN: ${F18} -fget-definition 8 20 23 -fparse-only %s >> %t;
! RUN: ${F18} -fget-definition 15 3 4 -fparse-only %s >> %t;
! RUN: ${F18} -fget-definition -fparse-only %s >> %t 2>&1;
! RUN: cat %t | ${FileCheck} %s
! CHECK:x:.*getdefinition01.f90, 6, 21-22
! CHECK:yyy:.*getdefinition01.f90, 6, 24-27
! CHECK:x:.*getdefinition01.f90, 14, 24-25
! CHECK:Invalid argument to -fget-definitions
