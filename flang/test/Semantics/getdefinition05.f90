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
! RUN: %f18 -fget-definition 9 5 6 -fsyntax-only %s | FileCheck --check-prefix=CHECK1 %s
! CHECK1: x:{{.*}}getdefinition05.f90, 7, 16-17
!! Outer y
! RUN: %f18 -fget-definition 11 7 8 -fsyntax-only %s | FileCheck --check-prefix=CHECK2 %s
! CHECK2: y:{{.*}}getdefinition05.f90, 5, 14-15
