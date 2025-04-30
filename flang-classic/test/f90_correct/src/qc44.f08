! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for Statement functions.

program test
  complex(16) :: cq1, cq2, c
  complex(16) :: rst
  real(16) :: i, j

  !define the statement functions:
  cq1(i, j) = cmplx(i, j, kind = 16)
  cq2(c) = (c - 2) * (c + c)

  rst = cq2(- cq1(1.0_16, 2.0_16))

  if (rst /= (-2.0_16, 16.0_16)) STOP 1

  print *, 'PASS'
end
