!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for the fix for LBOUND/UBOUND regression where the calling is in the
! parallel region.

program test
  implicit none
  integer :: x(7)
  call test_assumed_shp(10, 1, x)
  print *, "PASS"
contains
  subroutine test_assumed_shp(n, m, a)
    integer :: n, m
    integer :: a(n:)
    !$omp parallel
      if (lbound(a, 1) /= 10) STOP 1
      if (ubound(a, 1) /= 16) STOP 2
      if (lbound(a, m) /= 10) STOP 3
      if (ubound(a, m) /= 16) STOP 4
    !$omp end parallel
  end subroutine
end program
