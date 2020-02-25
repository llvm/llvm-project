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

! RUN: ${F18} -fget-symbols-sources -fparse-only %s 2>&1 | ${FileCheck} %s
! CHECK:x:.*getsymbols05.f90, 4, 14-15
! CHECK:y:.*getsymbols05.f90, 5, 14-15
! CHECK:x:.*getsymbols05.f90, 7, 16-17
