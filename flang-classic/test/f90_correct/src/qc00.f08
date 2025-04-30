! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for operation of quad precision and complex(16).

program test
  use check_mod
  integer, parameter :: m = 4, n = 8
  complex(16) :: resultc(m)
  real(16) :: result(n), expect(n)

  resultc(1) = 10.123456789012345678901234567890123456789_16 - (1.0, 2.0)
  resultc(2) = (1.0, 2.0) + 10.123456789012345678901234567890123456789_16
  resultc(3) = 10.123456789012345678901234567890123456789_16 - (3.643252352_8, 7.1242151253_8)
  resultc(4) = (-1.5131213124_8, 6.123125123_8) + 10.123456789012345678901234567890123456789_16

  result(1) = resultc(1)%re
  result(2) = resultc(1)%im
  result(3) = resultc(2)%re
  result(4) = resultc(2)%im
  result(5) = resultc(3)%re
  result(6) = resultc(3)%im
  result(7) = resultc(4)%re
  result(8) = resultc(4)%im

  expect(1) = 9.12345678901234567890123456789012386_16
  expect(2) = -2.0_16
  expect(3) = 11.1234567890123456789012345678901239_16
  expect(4) = 2.0_16
  expect(5) = 6.48020443701234548514657602834347023_16
  expect(6) = -7.12421512530000011764741429942660034_16
  expect(7) = 8.61033547661234564971545863852667941_16
  expect(8) = 6.12312512300000033604874261072836816_16

  call checkr16(result, expect, n)
end
