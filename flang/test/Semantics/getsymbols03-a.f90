! Tests -fget-symbols with INCLUDE
INCLUDE "Inputs/getsymbols03-b.f90"

program main
 use mm3
 integer :: x
 x = f
end program

! RUN: %f18 -fget-symbols-sources -fparse-only %s 2>&1 | FileCheck %s
! CHECK:mm3:{{.*}}getsymbols03-b.f90, 1, 8-11
! CHECK:f:{{.*}}getsymbols03-b.f90, 2, 12-13
! CHECK:main:{{.*}}getsymbols03-a.f90, 4, 9-13
! CHECK:x:{{.*}}getsymbols03-a.f90, 6, 13-14
