!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program zieee14
use ieee_arithmetic
real*4 a,b,c,c2
real*8 x,y,z,z2
logical lexp(4), lres(4)

a = sqrt(5.0)
x = sqrt(5.0d0)

b = exp(1.0) / 2.0
y = exp(1.0d0) / 2.0d0

c = ieee_rem(a,b)
z = ieee_rem(x,y)

c2 = a - b*2.0
z2 = x - y*2.0d0

lres(1) = (abs(c2 - c) .lt. 0.00001)
lres(2) = (abs(z2 - z) .lt. 0.000000001d0)
z = ieee_rem(a,y)
lres(3) = (abs(z2 - z) .lt. 0.00001d0)
z = ieee_rem(x,b)
lres(4) = (abs(z2 - z) .lt. 0.00001d0)

lexp = .true.
call check(lres, lexp, 4)

end
