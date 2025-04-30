! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for intrinsic real and imag for quad complex

program test
  implicit none
  integer, parameter :: n = 4
  integer :: j
  complex(16) :: c1, c2
  real(16) :: r, i, rs, is
  real(16) :: result(n), expect(n)
  r = -1.23456789012345678901234567890123456789_16
  i = 9.87654321098765432109876543210987654321_16
  c1%re = 1.23456789012345678901234567890123456789_16
  c1%im = -9.87654321098765432109876543210987654321_16

  c2%re = r
  c2%im = i

  expect(1) = 1.23456789012345678901234567890123456789_16
  expect(2) = -9.87654321098765432109876543210987654321_16
  expect(3) = -1.23456789012345678901234567890123456789_16
  expect(4) = 9.87654321098765432109876543210987654321_16

  result(1) = c1%re
  result(2) = c1%im
  result(3) = c2%re
  result(4) = c2%im

  do j = 1, n
    if (result(j) /= expect(j)) STOP j
  enddo

  print *, 'PASS'
end
