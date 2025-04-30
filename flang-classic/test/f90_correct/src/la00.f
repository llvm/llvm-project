** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Miscellaneous Scheduler and other bugs.

c    (1) Complex store problem.
c    (2) Dependency between CALL and cblock load.
c    (3) Multiplication of complex functions.
c    (4) Complex array subscripts
c    (5) CMPLX, DCMPLX intrinsics

	program p
	parameter(n = 8)
	complex c, arr(2)
	double complex darr(2)
	real rslts(n), expect(n)
	double precision dfunc
	real rfunc
	external dfunc, rfunc

	common kk

c  --- tests 1, 2:  Complex store problem:

	data expect(1), expect(2) / 4.0, 2.0 /
	data c / (2, 4) /
	c = cmplx(aimag(c), real(c))
	rslts(1) = real(c)
	rslts(2) = aimag(c)

c  --- test 3:  kk should be reloaded after call:

	data expect(3) / 3.0 /
	kk = 0
	call sub
	rslts(3) = kk

c  --- test 4 - 5:  Multiply two complex functions:

	data expect(4), expect(5) / -13.0, 52.0 /
	c = cos((1,2)) * sin((-2,-1))
	rslts(4) = int( 10 * real(c) )
	rslts(5) = int( 10 * aimag(c) )

c  --- test 6: number of times a subscript of a complex array is evaluated

	data expect(6) /1.0/
	rslts(6) = 0.0
	arr(ifunc(rslts(6))) = 0.0

c  ---  test 7: number of times the operand of cmplx is evaluated

	data expect(7) /1.0/
	rslts(7) = 0.0
	arr(1) = (1.0, 2.0)
	arr(2) = cmplx(rfunc(rslts(7))) * arr(1)

c  ---  test 8: number of times the operand of dcmplx is evaluated

	data expect(8) /1.0/
	rslts(8) = 0.0
	darr(1) = (1.0, 2.0)
	darr(2) = dcmplx(dfunc(rslts(8))) * darr(1)

c  --- check results:

	call check(rslts, expect, n)
	end

cccccccccccccccccccccccccccccc

	subroutine sub
	common kk
	kk = 3
	end

	integer function ifunc(x)
	x = x + 1
	ifunc = 1
	return
	end

	real function rfunc(x)
	x = x + 1
	rfunc = 1.0
	return
	end

	double precision function dfunc(x)
	x = x + 1
	dfunc = 1.0
	return
	end
