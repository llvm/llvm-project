!---------------------------------------------------------------------
! RUN: %flang_fc1 -emit-fir %s -o - 2>&1 | FileCheck %s
! XFAIL: *
!---------------------------------------------------------------------

program test_secnds
  real :: x
  x = secnds(1.0)
end program

! CHECK: not yet implemented: SECNDS

