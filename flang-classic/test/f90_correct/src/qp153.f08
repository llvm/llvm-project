! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for operation of quad precision.

program test
  integer, parameter :: n = 4
  real(16), parameter :: rst1(2) = (/1.0_16, 2.0_16/) * 2.0_16
  real(16), parameter :: rst2(2) = (/1.0_16, 2.0_16/) / 2.0_16

  if (rst1(1) /= 2.0_16) STOP 1
  if (rst1(2) /= 4.0_16) STOP 2
  if (rst2(1) /= 0.5_16) STOP 3
  if (rst2(2) /= 1.0_16) STOP 4

  print *, 'PASS'
end
