!RUN: %S/test_any.sh %s %t %f18
! Tests -fget-symbols-sources with COMMON.
program main
  integer :: x
  integer :: y
  common /x/ y
  x = y
end program

! EXEC: ${F18} -fget-symbols-sources -fparse-only %s 2>&1 | ${FileCheck} %s
! CHECK:x:.*getsymbols04.f90, 4, 14-15
! CHECK:y:.*getsymbols04.f90, 5, 14-15
! CHECK:x:.*getsymbols04.f90, 6, 11-12
