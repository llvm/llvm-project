! Tests -fget-symbols-sources with COMMON.
program main
  integer :: x
  integer :: y
  common /x/ y
  x = y
end program

! RUN: %f18 -fget-symbols-sources -fparse-only %s 2>&1 | FileCheck %s
! CHECK:x:{{.*}}getsymbols04.f90, 3, 14-15
! CHECK:y:{{.*}}getsymbols04.f90, 4, 14-15
! CHECK:x:{{.*}}getsymbols04.f90, 5, 11-12
