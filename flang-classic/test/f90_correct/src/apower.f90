!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This test case is test for power operator (int precision).
! The result of power using base -1 depends on the parity of the exponent.

program test
  integer :: i1, i2, res(6)
  integer(kind = 8) :: i2_8
  integer :: eres(6) = [1, 1, -1, -1, 0, 0]

  i2 = -8
  i1 = -1
  res(1) = i1 ** i2
  res(2) = i1 ** (-8)

  i2 = -7
  i1 = -1
  res(3) = i1 ** i2
  res(4) = i1 ** -7

  i2 = -7
  i1 = -2
  res(5) = i1 ** i2
  res(6) = i1 ** -7

  call check(res, eres, 6)
end
