! Tests -fget-symbols-sources with COMMON.
program main
  integer :: x
  integer :: y
  block
    integer :: x
    x = y
  end block
  x = y
end program

! RUN: %f18 -fget-symbols-sources -fparse-only %s 2>&1 | FileCheck %s
! CHECK:x:{{.*}}getsymbols05.f90, 3, 14-15
! CHECK:x:{{.*}}getsymbols05.f90, 6, 16-17
! CHECK:y:{{.*}}getsymbols05.f90, 4, 14-15
