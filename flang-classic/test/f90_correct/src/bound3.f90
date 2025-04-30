!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for LBOUND and UBOUND when the array is assumed-size.

program test
  implicit none
  integer :: x(2,3,4)
  call test_assumed_size(x)
  print *, "PASS"
contains
  subroutine test_assumed_size(a)
    integer :: a(2:5, 4:*)
    if (size(lbound(a)) /= 2 .or. any(lbound(a) /= [2, 4])) STOP 1
    if (any([lbound(a, 1), lbound(a, 2)] /= [2, 4])) STOP 2
    if (ubound(a, 1) /= 5) STOP 3
  end subroutine
end program
