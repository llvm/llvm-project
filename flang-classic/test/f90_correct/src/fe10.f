** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Intrinsic (double complex) names passed as arguments.
*   (all intrinsics which are allowed to be passed are tested once).

	parameter(n = 7)
	implicit real*8 (d)
	integer rslts(n), expect(n)

	integer ztoi
	intrinsic dconjg, cdsqrt, cdexp, cdlog, cdsin, cdcos
	intrinsic cdabs

	rslts(1) = ztoi((1.0d0, 2.0d0),  dconjg)
	rslts(2) = ztoi((2.0d0, -3.0d0), cdsqrt)
	rslts(3) = ztoi((-1.0d0, 1.0d0), cdexp)
	rslts(4) = ztoi((3.0d0, 4.0d0),  cdlog)
	rslts(5) = ztoi((3.0d0, 4.0d0),  cdsin)
	rslts(6) = ztoi((3.0d0, 4.0d0),  cdcos)

c  --- tests 7:  cdabs

	call sub4(rslts(7), cdabs)

c  --- check results:

	call check(rslts, expect, n)

	data expect /
     +       980, 1665, 201, 1618, 3583, -27073,
     +       5
     +              /
	end

c ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	
	integer function ztoi(c, f)
	double complex c, f, t
	external f
	
	t = f(c)
	ztoi = real(t) * 1000 + dimag(t) * 10
	end

	subroutine sub4(a, cdabs)
	external cdabs
	double precision cdabs
	integer a(*)

	a(1) = cdabs((3.1d0, 4.0d0))
	end
