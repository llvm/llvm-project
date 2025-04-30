** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Type conversion generics and intrinsics: INT, IFIX, IDINT,
*   REAL, FLOAT, SNGL, DBLE, DFLOAT, and CMPLX.

	program p
	implicit double precision (d), complex (c)
	integer rslts(12), expect(12)
	real rrslts(13:23), rexpect(13:23)
	double precision drslts(7), dexpect(7)
	complex crslts(9), cexpect(9)
	common /r/ rslts, rrslts, drslts, crslts
	common /e/ expect, rexpect, dexpect, cexpect

	integer*2 j3

	data expect / 2, 3, 5, 5, -5, 5, 0, 5, 10, 7, 4, 3 /
	data rexpect/ 2.5, -5.9, 3.0, -5.0, 3.0, 0.0,
     +                5.5, 2.0, -5.9, -5.5, 2.0       /
	data dexpect/ 99999D0, -3.0d0, 99996d0, 2.0d0,
     +                5.5d0, 5.9, 3.0d0             /
	data cexpect/ (1234.0, 0.0),   (2.0, 0.0),
     +                (3.0, 0.0),      (4.0, 1.0),
     +                (66000.0, -5.0), (3.0, 2.0),
     +                (1.0, -1.0),     (5.5, 20.0),
     +                (1.0, -1.0)                     /

	data i2, j3 / 2, 3/
	data xn59, x0, x1 / -5.9, 0.0, 1.0 /
	data d59, d55, d1, d2, d11/ 5.9D0, 5.5d0, 1.0d0, 2.0d0, 11d0 /
	data c31 / (3.0, 1.0) /

c ------------- tests 1 - 7:     integer, real ----> integer

	call sub1(int(i2), int(j3), rslts)
	rslts(3) = int(i2 + j3)
	rslts(4) = int(5.9)
	rslts(5) = int(xn59)
	rslts(6) = ifix(-xn59)
	rslts(7) = ifix(0.0001)

c ------------- tests 8 - 12:    double, complex ----> integer

	rslts(8) = int(d59)
	rslts(9) = idint(d59) + int(5.9d0)
	rslts(idint(10.5d0)) = 7
 	rslts(int(d11)) = int((4.0, 3.0))
 	rslts(12) = int( c31 )

c ------------- tests 13 - 18:   real, integer ----> real

	rrslts(13) = real(2.5)
	rrslts(14) = real(xn59)
	rrslts(15) = real(j3)
	rrslts(16) = float( -(i2 + j3) )
	rrslts(17) = float(j3)
	rrslts(18) = real(-1) + float(1)

c ------------- tests 19 - 23:   double, complex ----> real

	call sub2(sngl(d55), real(2.0d0), real(xn59), rrslts(19))
	rrslts(22) = real( -d55 )
	rrslts(23) = real(c31) + real((-1.0, 5.0))

c ------------- tests 24 - 29:   integer ----> double

	drslts(1) = dfloat(99999)
	drslts(2) = dfloat( -j3 )
	drslts(3) = dble( 99999 - j3 )

c ------------- tests 30 - 37:   real, double, complex ----> double

	drslts(4) = int( dble(12345.1) - 12343.0 )
        drslts(5) = dble( d55 )
	drslts(6) = dble(x0 - xn59)
	drslts(7) = dble(c31)

c ------------- tests 38 - 45:   CMPLX intrinsic with one argument:

	crslts(1) = cmplx(1234)
	crslts(2) = cmplx(x1) + x1
	crslts(3) = cmplx(d1 + d2)
	crslts(4) = cmplx(c31 + cmplx(x1))

c ------------- tests 46 - 55:   CMPLX intrinsic with two arguments:

	crslts(5) = cmplx(66000, -5)
	crslts(6) = cmplx(INT(j3), i2)
	crslts(7) = cmplx(1.0, -x1)
	crslts(8) = cmplx( d55, 2.0d1 )
	crslts(9) = cmplx(-1.0, -1.0) + cmplx(2.0)

c ------------- check results:

	call check(rslts, expect, 55)
	end

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

	subroutine sub1(i, j, a)
	integer a(2)

	a(1) = i
	a(2) = j
	i = 99
	j = 99
	end

	subroutine sub2(x, y, z, a)
	real x, y, z, a(3)

	a(1) = x
	a(2) = y
	a(3) = z
	end
