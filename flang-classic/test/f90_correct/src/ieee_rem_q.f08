! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

program test
  use ieee_arithmetic
  real(16), dimension(14) :: rslt
  real(16), dimension(9) :: ex
  real(16) :: c1, c2, c3, c4, x, y, z
  real(8) :: m, n
  real(4) :: a, b
  integer :: i

  c1 = 0.0_16
  c2 = 1.0_16
  c3 = -0.0_16
  c4 = -1.0_16
  a = sqrt(5.0)
  m = sqrt(5.0d0)
  x = sqrt(5.0_16)
  b = exp(1.0) / 2.0
  n = exp(1.0d0) / 2.0d0
  y = exp(1.0_16) / 2.0_16
  z = x - y * 2.0_16

  ex(1) = 0.00000000000000000000000000000000000_16
  ex(2) = -0.00000000000000000000000000000000000_16
  ex(3) = -0.00000000000000000000000000000000000_16
  ex(4) = -0.00000000000000000000000000000000000_16
  ex(5) = 0.00000000000000000000000000000000000_16
  ex(6) = 0.00000000000000000000000000000000000_16
  ex(7) = 0.00000000000000000000000000000000000_16
  ex(8) = 7.48463710820404818233528368378611716E-0002_16
  ex(9) = -7.48463710820404818233528368378611716E-0002_16

  rslt(1) = ieee_rem(c1, c2)
  rslt(2) = ieee_rem(c3, c2)
  rslt(3) = ieee_rem(c4, c2)
  rslt(4) = ieee_rem(-tiny(1.0_16), tiny(1.0_16))
  rslt(5) = ieee_rem(c2, tiny(1.0_16))
  rslt(6) = ieee_rem(tiny(1.0_16), tiny(1.0_16))
  rslt(7) = ieee_rem(huge(1.0_16), tiny(1.0_16))
  rslt(8) = ieee_rem(huge(1.0_16), 0.3_16)
  rslt(9) = ieee_rem(-huge(1.0_16), 0.3_16)
  rslt(10) = ieee_rem(a, y)
  rslt(11) = ieee_rem(x, b)
  rslt(12) = ieee_rem(m, y)
  rslt(13) = ieee_rem(x, n)
  rslt(14) = ieee_rem(x, y)

  do i = 1, 14
    if (i <= 9) then
      if (rslt(i) /= ex(i)) STOP i
    else
      if (abs(z - rslt(i)) / z >= 5e-33_16) STOP i
    endif
  enddo

  print *, 'PASS'
end
