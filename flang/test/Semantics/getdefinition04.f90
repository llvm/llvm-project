! Tests -fget-definition with COMMON block with same name as variable.
program main
  integer :: x
  integer :: y
  common /x/ y
  x = y
end program

! RUN: %flang_fc1 -fsyntax-only -fget-definition 6 3 4 %s | FileCheck %s
! CHECK: x:{{.*}}getdefinition04.f90, 3, 14-15
