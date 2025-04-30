! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for ieee_logb, ieee_rint, ieee_scalb with real*16

program ieee_test
  use ieee_arithmetic
  real*16 :: a, ral, rsa, rone
  real*16 :: aa1(140), aa2(140), aa3(140), aa4(140)
  logical :: ll1(140), ll2(140), le1(420), le2(420)
  logical :: luflow
  logical :: lfsav(5), lfset(5)

  lfset = .false.
  call ieee_get_halting_mode(ieee_all, lfsav)
  call ieee_set_halting_mode(ieee_all, lfset)

  call ieee_get_underflow_mode(luflow)
  call ieee_set_underflow_mode(.true.)

  rone = 1.0q0
  rlogb = 0.0q0
  a = rone
  do j = 1, 140
    ral = ieee_logb(a)
    aa1(j) = ral
    aa2(j) = rlogb
    i = int(ieee_rint(ral), kind = 8)
    rsa = ieee_scalb(rone, i)
    aa3(j) = rsa
    aa4(j) = a
    a = a * 0.5q0
    rlogb = rlogb - 1.0q0
  end do

  do j = 1, 140
    ll1(j) = aa1(j) .eq. aa2(j)
    ll2(j) = aa3(j) .eq. aa4(j)
  end do

  le1 = .false.
  le2 = .true.

  le1(1:140) = ll1
  le1(141:280) = ll2

  call ieee_set_underflow_mode(.false.)

  rone = 1.0q0
  rlogb = 0.0q0
  a = rone
  do j = 1, 140
    ral = ieee_logb(a)
    i = int(ieee_rint(ral), kind = 8)
    rsa = ieee_scalb(rone, i)
    aa3(j) = rsa
    aa4(j) = a
    a = a * 0.5q0
    rlogb = rlogb - 1.0q0
  end do

  do j = 1, 140
    ll2(j) = aa3(j) .eq. aa4(j)
  end do

  le2 = .true.
  le1(281:420) = ll2
  call check(le1, le2, 420)

  call ieee_set_underflow_mode(luflow)
  call ieee_set_halting_mode(ieee_all, lfsav)
end
