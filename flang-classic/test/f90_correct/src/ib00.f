** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Subprogram names as actual arguments (dummy procedures).

	parameter(n = 10)
	common /r/ rslts(n)
	integer rslts, expect(n)

	integer part2
	external iadd, isub, ctoi
	external f6
	external set3n, set5

c --- tests 1 - 5:

	call part1(iadd, ctoi, f6, set3n, set5)

c --- tests 6 - 10:

	rslts(n) = part2(iadd, isub)

c --- check results:

	call check(rslts, expect, iadd(n, 0))

	data expect / -4, 9, 6, 3, 5,
     +                5, 110, -90, 7, 10 /
	end

c---------------------------------

	subroutine part1(iaddx, ctoix, f6x, set3nx, set5x)
	parameter(n = 10)
	common /r/ rslts(n)
	integer rslts, expect(n)

	integer ctoix, f6x
	external  ctoix, set3nx

	rslts(1) = iaddx(2, iaddx(-10, 4))
	rslts(2) = 3 + ctoix((2.0, 4.0))
	rslts(3) = f6x()
	call set3nx(iaddx(1, 3))
	call set5x
	end

ccccccccccccccc

	integer function part2(iaddx, isubx)
	parameter(n = 10)
	common /r/ rslts(n)
	integer rslts, expect(n)

	external iaddx

	part2 = 7
	rslts(6) = isubx(part2, 2)
	part2 = 10
	call assign(7, iaddx, 10, 100)
	call assign(8, isubx, 10, 100)
	rslts(9) = isubx(iaddx(2, 5), isubx(10, 2)) + isubx(part2, 2)

	end

ccccccccccccccc

	function iadd(i, j)
	iadd = i + j
	end

ccccccccccccccc

	function isub(i, j)
	data i1 / 1 /
	isub = (i - j) / i1
	end

ccccccccccccccc

	function ctoi(c)
	complex c
	integer ctoi

	ctoi = real(c) + aimag(c)
	end

ccccccccccccccc

	function f6()
	implicit integer (f)
	f6 = 6
	return
	end

ccccccccccccccc

	subroutine set3n(nn)
	parameter(n = 10)
	common /r/ rslts(n)
	integer rslts, expect(n)
	rslts(nn) = 3
	end

ccccccccccccccc

	subroutine set5
	parameter(n = 10)
	common /r/ rslts(n)
	integer rslts, expect(n)
	rslts(5) = 5
	end

ccccccccccccccc

	subroutine assign(m, f, v1, v2)
	implicit integer (a-z)
	parameter(n = 10)
	common /r/ rslts(n)
	integer rslts, expect(n)

	rslts(m) = f(v1, v2)
	end
