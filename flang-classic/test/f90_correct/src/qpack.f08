! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test PACK intrinsic with quad-precision arguments

program main
  use check_mod
  use ieee_arithmetic
  real(16), dimension(3, 3, 3) :: array
  real(16) :: vector(6), res(6), expct(6), r(26), er(26)
  logical :: mas(3, 3, 3)
  array = 0.0_16
  vector = 0.0_16
  data expct / 0.999999999999999999999999999999999904_16, &
               1.99999999999999999999999999999999981_16,  &
               2.99999999999999999999999999999999961_16,  &
               1.00000000000000000000000000000000000_16,  &
               2.00000000000000000000000000000000000_16,  &
               3.00000000000000000000000000000000000_16 /

  do i = 1, 3
    array(1, i, 1) = ieee_next_after(real(i, kind = 16), 0.0_16)
    vector(i + 3) = i
  enddo

  mas = array /= 0
  res = pack(array, mas, vector)
  r(1:6) = res
  er(1:6) = expct
  res = pack(array, mas)
  r(7:9) = res(1:3)
  er(7:9) = expct(1:3)
  
  mas(1, 2, 1) = .false.
  res = pack(array, mas, vector)
  expct(2) = expct(3)
  expct(3) = 0.0_16
  r(10:15) = res
  er(10:15) = expct
  res(1:2) = pack(array, mas)
  r(16:18) = res(1:3)
  er(16:18) = expct(1:3)
  
  array(1, 3, 1) = huge(1.0_16)
  res = pack(array, mas, vector)
  expct(2) = 1.18973149535723176508575932662800702E+4932_16
  r(19:24) = res
  er(19:24) = expct
  res = pack(array, mas)
  r(25:26) = res(1:2)
  er(25:26) = expct(1:2)
  
  call checkr16(r, er, 26)
end
