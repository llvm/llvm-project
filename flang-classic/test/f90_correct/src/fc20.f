** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Intrinsics and generics: MAX and MIN - (no constant folding).

	implicit double precision (d)
	common rslts
	parameter(n = 26)
	integer rslts(n), expect(n)

	data i2, i5, in5 / 2, 5, -5 /
	data x2, x5, x15, xn5 / 2.0, 5.0, 1.5, -5.0 /
	data d2, d5, d15, dn5 / 2.0, 5.0, 1.5, -5.0 /

c --- tests 1 - 6: INTEGER MAX and MIN operations with 2 arguments:

	rslts(1) = max(i2, i5) + max(-1, in5)
	rslts(2) = max0(i5 + i2, 6) - min0(0, i5 * i2)
	rslts(3) = min(i2, -i5) * min(-i5, 0)
	rslts(4) = min(max(3, i2), i5 + i5)
c       use rslts(5) to count number of calls of function if:
	rslts(5) = 0
	rslts(min(i5+1, '7fffffff'x)) = max0(if(3), if(4))

c --- tests 7 - 12: REAL and DOUBLE PRECISION MAX and MIN:

	rslts(7) = ifix(max(x5, x2)) + ifix(amax1(-1.0, xn5))
	rslts(8) = amax1(x2, x15+1.0) + min(x15, -xn5)
	rslts(9) = amin1(0.0, xn5) * min(x2-x5, 2.35)

	rslts(10) = idint(min(d2, d5)) + idint(dmin1(dn5, -1.0d0))
	rslts(11) = dmin1(d15, d2+1.0) - max(-d15, dn5)
	rslts(12) = dmax1(1.5d0, d2-1) * max(dn5*(-d2), 0.0d0)

c --- tests 13 - 16: MAX and MIN operations which convert types:

	rslts(13) = ifix(amax0(i5, i2))
	rslts(14) = max1(2.5, x2) * 2.01
	rslts(15) = ifix(amin0(i5, i2))
	rslts(16) = 2.01 * min1(xn5/(-2.0), -xn5)

c --- tests 17 - 26: MAX and MIN operations with 3 or more arguments:

	rslts(17) = max(5, i2, -i2)
	rslts(18) = amax1(x2, 1+x5, 2.3, x5-x2)
	rslts(19) = max(d2, -d2, 2.5d0) * 2
	rslts(20) = amax0(3, 2, 1, i2)
	rslts(21) = max1(x2, x2, -5.0)

	rslts(22) = min0(-i2, i2, 5)
	rslts(23) = i2 * min(-xn5, 2.6, -xn5)
	rslts(24) = dmin1(5.0d0, 0.0d0, -d2)
	rslts(25) = if2(amin0(1, 2, -i2, -i5, -100))
	rslts(26) = min1(x2, -x2, -2.5) * 2

c --- check results:

	call check(rslts, expect, n)

c         --- tests 1 - 6:
	data expect / 4, 7, 25, 3, 2, -3,
c         --- tests 7 - 12:
     +                4, 4, 15, -3, 3, 15,
c         --- tests 13 - 16:
     +                5, 4, 2, 4,
c         --- tests 17 - 26:
     +                5, 6, 5, 3, 2, -2, 5, -2, 101, -4 /
	end


	integer function if(i)
	common rslts
	integer rslts(26)
	if = - i
	rslts(5) = rslts(5) + 1
	end

	integer function if2(x)
	if2 = ((-x) + 1.01)
	return
	end
