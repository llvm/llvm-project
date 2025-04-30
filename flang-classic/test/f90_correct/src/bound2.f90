!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for LBOUND and UBOUND when the array is assumed-rank.

program test
  implicit none
  integer :: x(2,3,4)
  call test_assumed_rank(x)
  print *, "PASS"
contains
  subroutine test_assumed_rank(a)
    integer :: a(..)
    if (size(lbound(a)) /= 3 .or. any(lbound(a) /= 1)) STOP 1
    if (size(ubound(a)) /= 3 .or. any(ubound(a) /= [2, 3, 4])) STOP 2
  end subroutine
end program
