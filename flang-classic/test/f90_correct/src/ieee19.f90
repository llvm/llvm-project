!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program testieee19
use ieee_arithmetic
real*8 a, ral, rsa, rone
real*8 aa1(1040), aa2(1040), aa3(1040), aa4(1040)
logical ll1(1040), ll2(1040), le1(1040), le2(1040)
logical luflow!, cg_support_underflow_control
logical lfsav(5), lfset(5)

lfset = .false.
call ieee_get_halting_mode(ieee_all, lfsav)
call ieee_set_halting_mode(ieee_all, lfset)

call ieee_get_underflow_mode(luflow)
call ieee_set_underflow_mode(.true.)

rone = 1.0d0
rlogb = 0.0d0
a = rone
do j = 1, 1040
  ral = ieee_logb(a)
  aa1(j) = ral
  aa2(j) = rlogb
  i = int(ieee_rint(ral))
  rsa = ieee_scalb(rone,i)
  aa3(j) = rsa
  aa4(j) = a
  a = a * 0.5d0
  rlogb = rlogb - 1.0d0
end do

do j = 1, 1040
  ll1(j) = aa1(j) .eq. aa2(j)
  ll2(j) = aa3(j) .eq. aa4(j)
end do

le1 = .true.
le2 = .true.

call check(ll1, le1, 1040)
call check(ll2, le2, 1040)

! We need to stop putting conditions in tests
!
! if (cg_support_underflow_control(a)) then

call ieee_set_underflow_mode(.false.)

rone = 1.0d0
rlogb = 0.0d0
a = rone
do j = 1, 1040
  ral = ieee_logb(a)
  i = int(ieee_rint(ral))
  rsa = ieee_scalb(rone,i)
  aa3(j) = rsa
  aa4(j) = a
  a = a * 0.5d0
  rlogb = rlogb - 1.0d0
end do

do j = 1, 1040
  ll2(j) = aa3(j) .eq. aa4(j)
end do

le2 = .true.

call check(ll2, le2, 1040)
! end if

call ieee_set_underflow_mode(luflow)
call ieee_set_halting_mode(ieee_all, lfsav)
end

! This function uses a "bug" in our oldcg implementation of exp
!logical function cg_support_underflow_control(a)
!use ieee_arithmetic
!real*8 a, b
!b = ieee_value(a,ieee_negative_inf)
!cg_support_underflow_control = ieee_is_normal(exp(b))
!return
!end

