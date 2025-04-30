! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for ieee_is_negative, ieee_is_normal, ieee_is_negative and
! ieee_unordered with real*16

program ieee_test
  use ieee_arithmetic
  real*16 :: a, b, c, x, y, z
  logical :: lexp(12), lres(12)
  logical :: lfsav(5), lfset(5)

  lfset = .false.
  call ieee_get_halting_mode(ieee_all, lfsav)
  call ieee_set_halting_mode(ieee_all, lfset)

  a = sqrt(5.0_16)
  x = sqrt(5.0q0)

  b = log(0.5_16)
  y = log(0.5q0)

  lres(1) = ieee_is_negative(b)
  lres(2) = ieee_is_negative(y)
  lres(3) = ieee_is_normal(a)
  lres(4) = ieee_is_normal(x)

  c = ieee_copy_sign(a, b)
  z = ieee_copy_sign(x, y)

  lres(5) = ieee_is_negative(c)
  lres(6) = ieee_is_negative(z)

  a = sqrt(c)
  x = sqrt(z)

  lres(7) = ieee_unordered(a, b)
  lres(8) = ieee_unordered(x, y)
  lres(9) = ieee_unordered(b, a)
  lres(10) = ieee_unordered(y, x)
  lres(11) = .not. ieee_is_normal(a)
  lres(12) = .not. ieee_is_normal(x)

  lexp = .true.
  call check(lres, lexp, 12)

  call ieee_set_halting_mode(ieee_all, lfsav)
end
