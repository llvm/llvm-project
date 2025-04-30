** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Intrinsics and generics: trigonometric and hyperbolic functions.

	program p
	implicit complex (c), double precision (d), double complex(z)
	parameter(n=55)
	integer rslts(n), expect(n), ctoi, dtoi
	parameter (d_dr=0.174532925199432957692D-1)
	parameter (r_dr=0.174532925199432957692E-1)
	parameter (d_rd=0.572957795130823208769D+2)
	parameter r_rd=0.572957795130823208769E+2

	dtoi(d) = d * 1000 + .499
	ctoi(c) = 1000 * (real(c) + aimag(c))
	ztoi(z) = 1000 * (real(z) + dimag(z))
	d_dtor(d) = d_dr*d
	d_rtod(d) = d_rd*d
	r_dtor(r) = r_dr*r
	r_rtod(r) = r_rd*r

	data x3, xx3 / 2 * 3.0 /,  d1, cx / 1.0d0, (1.0,2.0) /
	data zx/(1.0d0,2.0d0)/

c --- tests 1 - 4:   SIN

	rslts(1) = sin(.5) * 1000
	rslts(2) = dsin(d1) * 1000
	rslts(3) = dtoi( sin(dble(.5)) )
	rslts(4) = ctoi( csin( cx ) )

c --- tests 5 - 8:   COS

	rslts(5) = nint(cos(0.0))
	rslts(6) = dtoi( dcos(.34907d0) )
	rslts(7) = .001 + cos(x3 - xx3) * 1000
	rslts(8) = ctoi( ccos( cx ) )

c --- tests 9 - 11:  TAN

	rslts(9) = tan(1 / (x3 + 1)) * 1000
	rslts(10) = 1000 * tan(dble(x3) - 2)
	rslts(11) = dtan(.25d0) * 1000

c --- tests 12 - 14: ASIN

	rslts(12) = nint(asin(.47942) * 100)
	rslts(13) = nint(dasin(dble(x3) - 2) * 10)
	rslts(14) = asin(1.0d0) * 10

c --- tests 15 - 17: ACOS

	rslts(15) = nint(acos(1.0) * 100)
	rslts(16) = 100 * dacos(.93969 * 1d0) + .2
	rslts(17) = acos(1.0d0) + x3 + .01

c --- tests 18 - 20: ATAN

	rslts(18) = atan(0.0) * 100
	rslts(19) = dtoi( atan(.54630d0) )
	rslts(20) = dtoi( -datan(.54630d0) )

c --- tests 21 - 23: ATAN2

	rslts(21) = atan2(.54630, 1.0) * 1000 + .1
	rslts(22) = atan2(1.09260d0, x3-1d0) * 1000 + .1
	rslts(23) = datan2(.25534d0/2, dble(x3/6)) * 1000 + .1

c --- tests 24 - 26: SINH

	rslts(24) = sinh(3.0)
	rslts(25) = sinh( dble(x3) + 1)
	rslts(26) = dsinh(dble(x3) + 1)

c --- tests 27 - 29: COSH

	rslts(27) = -cosh(2.09)
	rslts(28) = cosh(dble(2.09))
	rslts(29) = dcosh(1d1)

c --- tests 30 - 32: TANH

	rslts(30) = 1 / tanh(.02d0)
	rslts(31) = tanh(x3) * 1000
	rslts(32) = dtanh( dble(x3) ) * 1000

c --- tests 33 - 36:   SIN

	rslts(33) = sind(r_rtod(.5)) * 1000
	rslts(34) = dsind(d_rtod(d1)) * 1000
	rslts(35) = dtoi( sind(dble(r_rtod(.5))) )
	rslts(36) = ztoi( cdsin( zx ) )

c --- tests 37 - 40:   COS

	rslts(37) = nint(cosd(0.0))
	rslts(38) = dtoi( dcosd(d_rtod(.34907d0)) )
	rslts(39) = .001 + cosd(r_rtod(x3 - xx3)) * 1000
	rslts(40) = ztoi( cdcos( zx ) )

c --- tests 41 - 43:  TAN

	rslts(41) = tand(r_rtod(1 / (x3 + 1))) * 1000
	rslts(42) = 1000 * tand(d_rtod(dble(x3) - 2))
	rslts(43) = dtand(d_rtod(.25d0)) * 1000

c --- tests 44 - 46: ASIN

	rslts(44) = nint(r_dtor(asind(.47942)) * 100)
	rslts(45) = nint(d_dtor(dasind(dble(x3) - 2)) * 10)
	rslts(46) = d_dtor(asind(1.0d0)) * 10

c --- tests 47 - 49: ACOS

	rslts(47) = nint(r_dtor(acosd(1.0)) * 100)
	rslts(48) = 100 * d_dtor(dacosd(.93969 * 1d0)) + .2
	rslts(49) = d_dtor(acosd(1.0d0)) + x3 + .01

c --- tests 50 - 52: ATAN

	rslts(50) = r_dtor(atand(0.0)) * 100
	rslts(51) = dtoi( d_dtor(atand(.54630d0)) )
	rslts(52) = dtoi( -d_dtor(datand(.54630d0)) )

c --- tests 53 - 55: ATAN2

	rslts(53) = r_dtor(atan2d(.54630, 1.0)) * 1000 + .1
	rslts(54) = d_dtor(atan2d(1.09260d0, x3-1d0)) * 1000 + .1
	rslts(55) = d_dtor(datan2d(.25534d0/2, dble(x3/6))) * 1000 + .1

c --- check results:

	call check(rslts, expect, n)

	data expect / 479, 841, 479, 5125,
     +                1, 940, 1000, -1019,
c           --- tests 9 - 14:
     +                255, 1557, 255, 50, 16, 15,
c           --- tests 15 - 20:
     +                0, 35, 3, 0, 500, -499,
c           --- tests 21 - 26:
     +                500, 500, 250, 10, 27, 27,
c           --- tests 27 - 32:
     +                -4, 4, 11013, 50, 995, 995,
c           --- tests 33 - 55 copy 1-23
     +		 479, 841, 479, 5125,
     +           1, 940, 1000, -1019,
     +           255, 1557, 255, 50, 16, 15,
     +           0, 35, 3, 0, 500, -499,
     +           500, 500, 250 /
	end
