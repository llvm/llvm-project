** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Intrinsics and generics - AINT, DINT, ..., CABS.

	parameter(n = 23)
	integer rslts(n), expect(n)
	real  rrslts(n), rexpect(n)
	equivalence (rslts, rrslts), (expect, rexpect)

	double precision d3, d349, d36
	complex c43

	data i234 / 234 /
	data x349, xn39 / 3.49, -3.9 /
	data d3, d349, d36 / 3d0, 3.49d0, 3.6d0 /
	data c43  / (4.0, 3.0) /

c ---- tests 1 - 5:  AINT, DINT:

	data (rexpect(i), i = 1, 5) /2.0, -3.0, -2.0, 3.0, 2.0 /
	rrslts(1) = aint(2.1)
	rrslts(2) = aint(xn39)
	rrslts(3) = sngl( aint(-2.9D0) )
	rrslts(4) = aint(d36)
	rrslts(5) = sngl( dint(2.0d0) )

c ---- tests 6 - 13: ANINT, DNINT, NINT, IDNINT:

	data (rexpect(i), i = 6, 9) / 3.0, -4.0, 249653.0, 3.0 /
	rrslts(6) = anint( 2.5 )
	rrslts(7) = anint( xn39 )
	rrslts(8) = sngl( anint(249653.1d0) )
	rrslts(9) = sngl( dnint(d349) )
	
	data (expect(i), i = 10, 13) / 23, -4, 4, 4 /
	rslts(10) = nint(22.5)
	rslts(11) = nint(xn39)
	rslts(12) = nint(d36 + .41)
	rslts(13) = idnint(.39 + d36)

c ---- tests 14 - 23: IABS, ABS, DABS, CABS:

	data (expect(i), i = 14, 16) / 5, 234, 234 /
	rslts(14) = iabs(-5)
	rslts(15) = iabs(i234)
	rslts(16) = abs( - i234 )

	data (rexpect(i), i = 17, 21) /
     +        23.47, 3.49, 2.341d-3, 2.0, +2.3d0 /
	rrslts(17) = abs(-23.47)
	rrslts(18) = abs(x349)
	rrslts(19) = sngl( abs(2.341d-3) )
	rrslts(20) = sngl( abs(1.0 - d3) )
	rrslts(21) = dabs(-2.3d0)

	data expect(22), expect(23) / 130, 50 /
	rslts(22) = nint(10 * cabs((-12.0, 5.0)))
	rslts(23) = nint(10 * abs( c43 ))

c ---- check results:

	call check(rslts, expect, n)

	end
