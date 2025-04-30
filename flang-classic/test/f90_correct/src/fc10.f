** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Intrinsics & generics - MOD, SIGN, DIM, DPROD.

	program p
	parameter (n = 32)
	integer rslts(n), expect(n)
	double precision d2, dn5

	data i0, i2, i5, i19, i7 / 0, 2, 5, 19, 7 /
	data x2, x4, x7 / 2.0, 4.0, 7.0 /
	data d2, dn5    / 2.0d0, -5.0d0 /

c --- tests 1 - 5: INTEGER MOD operation:

	rslts(1) = mod('7fffffff'x, 2)
	rslts(2) = mod('7ffffffe'x, i2)
	rslts(3) = mod(i2, i2) + mod(i19, 5) + mod(0, i19)
	rslts(4) = i2 * mod(i0, i19+7) + (i19 - mod(i19, i2))
	rslts(5) = mod(-7, 5) +  mod(-i7, 2 * i2 + 1)

c --- tests 6 - 10: REAL MOD operation:

	rslts(6) = mod(4.0, 2.0) - mod(-x4, x2)
	rslts(7) = amod(x2, x2+x2)
	rslts(8) = ifix( amod(x7, x4) )
	rslts(9) = amod(2.341, float(2)) * 100
	rslts(10) = 100 * mod(2.341, float(i2))

c --- tests 11 - 13: DOUBLE PRECISION MOD operation:

	rslts(11) = idint(dmod(5.0d0, 2.0d0))
	rslts(12) = idint( mod(dn5, -d2) )
	rslts(13) = dn5 + dmod(d2, d2)

c --- tests 14 - 17: INTEGER SIGN operation:

	rslts(14) = isign(3, 4) + isign(-8, -1)
	rslts(15) = sign(99, 0) + sign(-6, 1)
	rslts(sign(16, i2)) = isign(i2, 1)
	rslts(17) = isign(-i2, i19*i2) * sign(i19, -i2)

c --- tests 18 - 22: REAL and DOUBLE PRECISION SIGN operation:

	rslts(18) = sign(2.0, 4.0) + sign(8.0, -1.0)
	rslts(19) = sign(x7, -x4) * sign(x2*x2, x4)

	rslts(20) = idint( dsign(2.0d0, -1.0d0) )
	rslts(21) = idint( sign(dn5, d2 - d2) )
	rslts(22) = 0

c --- tests 23 - 29: DIM operation:

	rslts(23) = idim(i5, i2) + i19
	rslts(24) = dim(i2, i5) * i5
	rslts(25) = dim(i5*i2, i2 + i2)

	rslts(26) = dim(x7, x4)
	rslts(27) = 2.0 + dim(x4, x4+x4)

	rslts(28) = idint( ddim(dn5, d2) )
	rslts(29) = idint( dim(3.0d0, dn5 * 2.0) + 1)

c --- tests 30 - 32: DPROD operation:

	rslts(30) = idint(dprod(x2, x2))
	rslts(31) = dprod(-3.0, 2.0)
	rslts(32) = dprod(x2*x2, x4+x4)

c --- check results:

	call check(rslts, expect, n)

	data expect / 1, 0, 4, 18, -4,
     +                0, 2, 3, 34, 34,
     +                1, -1, -5,
     +                -5, 105, 2, -38, 
c        tests 18 - 22:
     +                -6, -28, -2, 5, 0,
c        tests 23 - 29:
     +                22, 0, 6, 3, 2, 0, 14,
     +                4, -6, 32              /
	end
