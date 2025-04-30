** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   ENTRY statements in function subprograms.
*   (non-identical return types only partially tested).

c   Regarding entries whose return types are different from the main
c   function, the only case tested is when the main function and all
c   entries are either integer, logical, or real (1 word types).

	parameter(n = 23)
	integer rslts(n), expect(n)
	character*4 crslts(n)
	equivalence(crslts, rslts)

	integer xiadd, yisub, aneg
	complex cp1, xcp1, c
	real*8 dp1, d8
	logical lneg
	character*4 cat, rcat, xcatz
	character alpha*2, alpha2*3, alphan*5, alphan2*1

	data i2 / 2 /

c --- tests 1 - 6:    INTEGER entries:

	rslts(1) = i7()
	rslts(2) = xiadd(3, -8)
	rslts(3) = iadd(i2, i2)
	rslts(4) = yisub(i2, 3)
	rslts(5) = i8()
	rslts(6) = ip1(i2)

c --- tests 7 - 10:   COMPLEX entries:

	c = cp1(3, -8)
	rslts(7) = real(c)
	rslts(8) = aimag(c)
	c = xcp1(i2, 3)
	rslts(9) = real(c)
	rslts(10) = aimag(c)

c --- tests 11 - 12:  DOUBLE PRECISION entries:

	rslts(11) = dp1(2.35) * 3
	rslts(12) = d8() * i2

c --- tests 13 - 15:  mixed entry return types:

	rslts(13) = aneg(1, i2)
	if (lneg(2, .false.))  rslts(14) = 3
	rslts(15) = 2 * xneg(3, 6.55)

c --- tests 16 - 19:  constant length character entries:

	crslts(16) = cat('a', 'bcd')
	crslts(17) = rcat('xy', 'zw')
	crslts(18) = xcatz( '.' )
	crslts(19) = cat('123', 'mn')

c --- tests 20 - 23:  passed length character entries:

	crslts(20) = alpha()
	crslts(21) = alphan(6)
	crslts(22) = alpha2()
	crslts(23) = alphan2(3)

c --- check results:

	call check(rslts, expect, n)

	data expect / 7, -5, 4, -1, 8, 3,
     +                4, -7, 3, 4,
     +                10, 16,
     +                -2, 3, -13,
     +                'abcd', 'zwxy', '.z  ', '123m',
     +                'ab  ', 'fghi', 'abc ', 'c   '   /
	end

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCC --- INTEGER entries:

	integer function i7()
	implicit integer (x)
	integer yisub, v2

	i7 = 7
	return

	entry xiadd(i, j)
	entry iadd(j, i)
	xiadd = i + j
	return

	entry yisub(j, v2)
	yisub = j - v2
	return

	entry i8
	yisub = 8
	return

	entry ip1(i)
	i7 = i
	yisub = i7 + 1
	end

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCC --- COMPLEX entries:

	function cp1(i, j)
	entry xcp1(i, j)
	implicit complex (c)
	complex xcp1

	cp1 = cmplx(i, j)
	xcp1 = xcp1 + (1, 1)
	end

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCC --- DOUBLE PRECISION entries:

	real*8 function dp1(x)
	double precision d8

	dp1 = x
10	dp1 = dp1 + 1
	return

	entry d8
	d8 = 7d0
	goto 10
	end

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCC --- mixed INTEGER, REAL, LOGICAL entries:

	integer*4 function aneg(flag, j)
	entry lneg(flag, larg)
	implicit logical (l)
	integer flag
	entry xneg(flag, z)

	if (flag .eq. 1)  aneg = -j
	if (flag .eq. 2)  lneg = .not. larg
	if (flag .eq. 3)  xneg = -z
	end

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCC --- constant length CHARACTER entries:

	function cat(a, b)
	entry rcat(b, a)
	character*4 cat, rcat, a*(*), b*(*), xcatz

	rcat = a // b
	return

	entry xcatz(b)
	cat = b // 'z'
	end

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCC --- passed length character entries:

	character*(*) function alpha()
	entry alpha2
	character*(*) alpha2, alphan, alphan2
	character*26 alph, tmp

	data tmp / 'abcdefghijklmnopqrstuvwxyz' /
	alph = tmp
	goto 20

	entry alphan(i)
	entry alphan2(i)

	alph = tmp(i:)

20	alpha = alph
	return
	end
