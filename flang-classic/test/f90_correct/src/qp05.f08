! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for comparison operation take quad precision

program test
  integer, parameter :: n = 2
  logical :: r1 = 1.1_16 > 1.2_16
  logical :: r2
  integer :: result(n), expect(n)
  expect = 1
  result = 0

  r2 = 621424.2346523454241479464_16 <= 621424.234652345424147954_16
  if (r1 .eq. .false.) result(1) = 1
  if (r2 .eq. .true.) result(2) = 1
  call check(result, expect, n)

end program
