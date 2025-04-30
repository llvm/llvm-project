! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for ieee_copy_sign, ieee_is_normal, ieee_is_negative and
! ieee_unordered with real*16

program ieee_test
  use ieee_arithmetic
  implicit none
  real*16 :: a, b, c, &
             plus_inf, minus_inf, nan_num
  logical :: res(16), expct(16)

  res = .false.
  a = 0.0q0
  plus_inf = 1.0q0 / a
  minus_inf = -1.0q0 / a
  nan_num = 0.0q0 / a

  b = ieee_copy_sign(plus_inf, -1.0q0)
  c = ieee_copy_sign(minus_inf, 1.0q0)
  res(1) = b .eq. minus_inf
  res(2) = c .eq. plus_inf

  res(3) = .not. ieee_is_normal(plus_inf)
  res(4) = .not. ieee_is_normal(minus_inf)
  res(5) = .not. ieee_is_normal(nan_num)

  res(6) = .not. ieee_is_negative(plus_inf)
  res(7) = ieee_is_negative(minus_inf)
  res(8) = .not. ieee_is_negative(nan_num)
  res(9) = ieee_is_negative(b)
  res(10) = .not. ieee_is_negative(c)

  res(11) = .not. ieee_unordered(0.0q0, plus_inf)
  res(12) = .not. ieee_unordered(0.0q0, minus_inf)
  res(13) = ieee_unordered(0.0q0, nan_num)
  res(14) = .not. ieee_unordered(plus_inf, minus_inf)
  res(15) = ieee_unordered(plus_inf, nan_num)
  res(16) = ieee_unordered(minus_inf, nan_num)

  expct = .true.
  call check(res, expct, 16)
end program
