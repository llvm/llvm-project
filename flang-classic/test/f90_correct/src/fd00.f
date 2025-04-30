** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Intrinsics and generics: SQRT, EXP, LOG, LOG10.

	program p
	implicit double precision (d)
	parameter(n = 21)
	integer rslts(n), expect(n)

	data d25, d36 / 25d0, 36d0 /
	data x5       / 5.0 /

c --- tests 1 - 5:    SQRT

	rslts(1) = sqrt(25.0) + .0001
	rslts(2) = .0001 + dsqrt(d36)
	rslts(3) = idnint( sqrt(d25) )
	rslts(4) = real(csqrt((-3.0, 4.0))) * 100 + .0001
	rslts(5) = nint( aimag(sqrt((-3.0, 4.0))) )

c --- tests 6 - 9:    EXP

	rslts(6) = 1000 * exp(1.0)
	rslts(7) = dexp(-1.0d0) * 1000
	rslts(8) = cexp( clog( (11.1, -13.1) ) )
	rslts(9) = aimag( exp( clog( (11.1, -13.1) ) ) )

c --- tests 10 - 13:  LOG

	rslts(10) = alog(x5 + x5) * 1000
	rslts(11) = dlog(d25 - 21) * 1000
	rslts(12) = clog( (10.0, 0.0) ) * 1000
	rslts(13) = log ( exp(-17.1d0) )

c --- tests 14 - 16:  LOG10

	rslts(14) = nint( log10(2 * x5) )
	rslts(15) = 1000 * alog10( float(2) )
	rslts(16) = dlog10( 1.001d25 )

c --- tests 17 - 18:    CDSQRT

	rslts(17) = real(cdsqrt((-3.0d0, 4.0d0))) * 100 + .0001
	rslts(18) = nint( dimag(sqrt((-3.0d0, 4.0d0))) )

c --- tests 19 - 20:    CDEXP

	rslts(19) = cdexp( cdlog( (11.1d0, -13.1d0) ) )
	rslts(20) = dimag( exp( cdlog( (11.1d0, -13.1d0) ) ) )

c --- tests 21 :  LOG

	rslts(21) = cdlog( (10.0d0, 0.0d0) ) * 1000


c --- check results:

	call check(rslts, expect, n)

	data expect / 5, 6, 5, 100, 2,
     +                2718, 367, 11, -13,
     +                2302, 1386, 2302, -17,
     +                1, 301, 25,
     +		      100,2,11,-13,2302 /
     /
	end
