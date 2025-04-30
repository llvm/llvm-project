!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program zieee13
use ieee_arithmetic
real*4 a,b,c
real*8 x,y,z
logical lexp(14), lres(14)
logical lfsav(5), lfset(5)

lfset = .false.
call ieee_get_halting_mode(ieee_all, lfsav)
call ieee_set_halting_mode(ieee_all, lfset)

a = sqrt(5.0)
x = sqrt(5.0d0)

b = log(0.5)
y = log(0.5d0)

lres(1) = ieee_is_finite(a)
lres(2) = ieee_is_finite(x)
lres(3) = ieee_is_negative(b)
lres(4) = ieee_is_negative(y)
lres(5) = ieee_is_normal(a)
lres(6) = ieee_is_normal(x)
c = ieee_copy_sign(a,b)
z = ieee_copy_sign(x,y)
lres(7) = ieee_is_negative(c)
lres(8) = ieee_is_negative(z)
a = sqrt(c)
x = sqrt(z)
lres(9) = ieee_is_nan(a)
lres(10) = ieee_is_nan(x)
lres(11) = ieee_unordered(a,b)
lres(12) = ieee_unordered(x,y)
lres(13) = ieee_unordered(b,a)
lres(14) = ieee_unordered(y,x)

lexp = .true.
call check(lres, lexp, 14)

call ieee_set_halting_mode(ieee_all, lfsav)
end
