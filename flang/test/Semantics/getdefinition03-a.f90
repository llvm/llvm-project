! Tests -fget-definition with INCLUDE

INCLUDE "getdefinition03-b.f90"

program main
 use m3
 integer :: x
 x = f
end program

! RUN: ${F18} -fget-definition 8 6 7 -fparse-only %s > %t;
! RUN: ${F18} -fget-definition 8 2 3 -fparse-only %s >> %t;
! RUN: cat %t | ${FileCheck} %s;
! CHECK:f:.*getdefinition03-b.f90, 2, 12-13
! CHECK:x:.*getdefinition03-a.f90, 7, 13-14
