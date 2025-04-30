! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test conversion to complex type with the CMPLX intrinsic

program test
  integer, parameter :: n = 7
  integer :: i
  complex(16) :: result(n), expect(n)
  integer(1) :: i_1 = 7_1
  integer(2) :: i_2 = 77_2
  integer(4) :: i_4 = 7777_4
  integer(8) :: i_8 = 7777_8
  real(4) :: r_4 = 1.234567_4
  real(8) :: r_8 = 1.234567890123456789_8
  real(16) :: r_16 = 1.23456789012345678901234567890123456789_16

  expect(1) = (7.0_16, 0.0_16)
  expect(2) = (77.0_16, 0.0_16)
  expect(3) = (7777.0_16, 0.0_16)
  expect(4) = (7777.0_16, 0.0_16)
  expect(5) = (1.234567_4, 0.0_16)
  expect(6) = (1.234567890123456789_8, 0.0_16)
  expect(7) = (1.23456789012345678901234567890123456789_16, 0.0_16)

  result(1) = cmplx(i_1, kind=16)
  result(2) = cmplx(i_2, kind=16)
  result(3) = cmplx(i_4, kind=16)
  result(4) = cmplx(i_8, kind=16)
  result(5) = cmplx(r_4, kind=16)
  result(6) = cmplx(r_8, kind=16)
  result(7) = cmplx(r_16, kind=16)

  do i = 1, n
    if (result(i) /= expect(i)) STOP i
  enddo

  print *, 'PASS'
end
