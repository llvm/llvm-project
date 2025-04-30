** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Statement functions.  Argumentless and non-character
*   statement functions only are tested.

	call mysub(3)
	end

	subroutine mysub(d3)
	implicit logical (l)
	integer d3, rslts(21), expect(21)
	integer TRUE
	parameter (TRUE = -1)
	dimension lrslts(21)
	equivalence ( lrslts, rslts)
	integer*2 h, i7, hf
	double precision df, df2, df3
	complex cf, cf2
	logical la(4)
	logical*1 lf3

c --- define the statement functions:

	if() = 22
	x() = real(d3)
	lf() = (d3 .lt. 3) .or. d3 .gt. 3

11	format(i3)
	if2() = d3

	entry myentry
	h() = i7
	hf() = d3 * 5

	data i7 / 7 /, la / .true., .false., .true., .false. /
	cos() = 2.6
	df() = cos()
	df2() = 2.6d0 + dble(if())
	df3() = -5
	if3() = d3 + (i7 * d3 * 2) - 10
	cf() = cmplx(d3, d3)
	cf2() = (-1.0, -2.0) * 2
	lf2() = i7 .eq. d3 + 4
	lf3() = la(d3)
	max0() = iadd(if2(), int(i7))

c --- tests 1 - 6:    INTEGER and INTEGER*2 statement functions:

	rslts(1) = if()
	rslts(2) = d3 + if()
	rslts(if2()) = if2() * 2
	rslts(4) = iadd(int(h()), if2()) * hf()
	rslts(5) = 2 * if3()
	rslts(6) = float(max0()) * 2.11

c --- tests 7 - 11:   REAL and DOUBLE PRECISION statement functions:

	rslts(7) = amax1(x(), cos())
	rslts(8) = 2 * cos()
	rslts(9) =  idnint(df())
	rslts(10) = df2()
	rslts(11) = -df3()

c --- tests 12 - 13:  COMPLEX statement functions:

	rslts(12) = cf() + cf2()
	rslts(13) = aimag(cf() * cf2())

c --- tests 14 - 21:  LOGICAL statement functions:

	lrslts(14) = lf()
	lrslts(15) =  lf2()
	lrslts(16) = lf3()

	do 10 i = 17, 21
10		rslts(i) = 0

	if (lf())  rslts(17) = 2
	if (.not. lf()) rslts(18) = 2
	if (lf3() .and. la(1))   rslts(19) = 2
	if (lf() .or. .not. lf2() .or. lf3())  rslts(20) = 2
	if (lf2() .and. lf())  rslts(21) = 2

c --- check results:

	call check(rslts, expect, 21)

	data expect / 22, 25, 6, 150, 70, 21,
     +                3, 5, 3, 24, 5,
     +                1, -18,
     +                0, TRUE, TRUE, 0, 2, 2, 2, 0   /
	end

	integer function iadd(i, j)
	iadd = i + j
	end
