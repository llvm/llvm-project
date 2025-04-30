!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program testieee19flushz
use ieee_arithmetic
real*8 a, ral, rone
real*8 aa1(1040), aa2(1040)
logical ll1(1040), ll2(1040), le1(1040)
logical luflow
logical lfsav(5), lfset(5)

lfset = .false.
call ieee_get_halting_mode(ieee_all, lfsav)
call ieee_set_halting_mode(ieee_all, lfset)

call ieee_get_underflow_mode(luflow)
call ieee_set_underflow_mode(.false.)

rone = 1.0d0
rlogb = 0.0d0
a = rone
do j = 1, 1040
  ral = ieee_logb(a)
  aa1(j) = ral
  if (j .le. 1023) then
    aa2(j) = rlogb
  else
    aa2(j) = ieee_value(rlogb, ieee_negative_inf)
  end if
  i = int(ieee_rint(ral))
  a = a * 0.5d0
  rlogb = rlogb - 1.0d0
end do

do j = 1, 1040
  ll1(j) = aa1(j) .eq. aa2(j)
end do

le1 = .true.

call check(ll1, le1, 1040)

call ieee_set_underflow_mode(luflow)
call ieee_set_halting_mode(ieee_all, lfsav)
end

