! Tests -fget-symbols with INCLUDE
INCLUDE "Inputs/getsymbols03-b.f90"

program main
 use mm3
 integer :: x
 x = f
end program

! RUN: %flang_fc1 -fsyntax-only -fget-symbols-sources %s 2>&1 | FileCheck %s
! CHECK:f:{{.*}}getsymbols03-b.f90, 2, 12-13
! CHECK:main:{{.*}}getsymbols03-a.f90, 4, 9-13
! CHECK:mm3:{{.*}}getsymbols03-a.f90, 5, 6-9
! CHECK:x:{{.*}}getsymbols03-a.f90, 6, 13-14
