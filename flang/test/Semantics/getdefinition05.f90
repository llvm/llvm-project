!RUN: %S/test_any.sh %s %flang %t
! Tests -fget-symbols-sources with BLOCK that contains same variable name as 
! another in an outer scope.
program main
  integer :: x
  integer :: y
  block
    integer :: x
    integer :: y
    x = y
  end block
  x = y
end program

!! Inner x
! EXEC: ${F18} -fget-definition 10 5 6 -fparse-only %s > %t;
! CHECK:x:.*getdefinition05.f90, 8, 16-17
!! Outer y
! EXEC: ${F18} -fget-definition 12 7 8 -fparse-only %s >> %t;
! CHECK:y:.*getdefinition05.f90, 6, 14-15
! EXEC: cat %t | ${FileCheck} %s;
