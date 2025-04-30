** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Relational operations (EQ - LT) for numeric types, requiring
*   type conversion of one operand, including constant folding.

	program p
	implicit logical (p)
	integer TRUE
	parameter (TRUE = -1)
	parameter(n=36)
	integer rslts(27:36), expect(n)
	logical lrslts(n)
	equivalence (rslts, lrslts(27))

	complex c12
	double precision d1
	logical t, f
	parameter(p1 = 2.0 .le. 2d0, p2 = -1.5d0 .ne. -1.5)
	parameter(p3 = (-1.0, -1.0) .ne. -1.0 )
	parameter(p4 = 1.0d0 .eq. (1.0, 0.0))

	data i0, i1, x1, x2, d1, c12 / 0, 1, 1.0, 2.0, 1.0d0, (1.0, 2.0) /
	data t, f, i2 / .true., .false., 2 /

c --- tests 1 - 6:    INTEGER/REAL comparisons:

	lrslts(1) = i1 .eq. x1
	lrslts(2) = 3 .gt. -x1
	lrslts(3) = i1 + 3 .lt. 4.0
	lrslts(4) = x2 * x2 .ge. i1 * 4
	lrslts(5) = 3.1 .le. 3
	lrslts(6) = xfn5(x2 .ge. 2) .ne. -5

c --- tests 7 - 10:   INTEGER/DOUBLE PRECISION comparisons:

	lrslts(7) = 3 .lt. 3.1d0 .and. t
	lrslts(8) = f .or. i1 .ge. d1 + 1
	lrslts(9) = d1 + i0 .eq. i1 * i1 .eqv. t
	lrslts(10) = -3.0d0 .gt. i1 + i1

c --- tests 11 - 14:  INTEGER/COMPLEX comparisons:

	lrslts(11) = 1 .eq. (1.0, 0.0)
	lrslts(12) = i2-2 .eq. (1.0, 0.0)
	lrslts(13) = (-1.0,0.0) .ne. -1
	lrslts(14) = .not. .not. c12 .ne. 1

c --- tests 15 - 18:  REAL/DOUBLE PRECISION comparisons:

	lrslts(15) = p1
	lrslts(16) = -x2 .gt. -d1
	lrslts(17) = p2
	lrslts(18) = d1 + d1 .lt. 2.001

c --- tests 19 - 22:  REAL/COMPLEX comparisons:

	lrslts(19) = 1.0 .eq. (2.0, 0.0)
	lrslts(20) = x2 .eq. (c12*2) - (0.0, 4.0)
	lrslts(21) = p3
	lrslts(22) = c12 * conjg(c12) .ne. x2 + 3.0

c --- tests 23 - 26:  DOUBLE PRECISION/COMPLEX comparisons:

	lrslts(23) = p4
	lrslts(24) = d1 * d1 .eq. c12
	lrslts(25) = (-2.0, 0.0) .ne. -2.0d0
	lrslts(26) = 2.0d0 .ne. c12 + 1.0

c --- tests 27 - 32:  comparisons in IF conditions - constant folding:

	data rslts / 10 * 0 /

	if (3 .eq. 3.0)   rslts(27) = 1
	if ((1.0, 0.0) .ne. 1)  rslts(28) = 1
	if (-5 .lt. -4.9d0)     rslts(29) = 1
	if (2d0 .ge. 2.1 .and. .true.) rslts(30) = 1
	if (-2.0 .eq. (-2.0, 0.0))  rslts(31) = 1
	if ((1.0, 0.0) .eq. 2.0)    rslts(32) = 1

c --- tests 33 - 36:  non-constant comparisons:

	if (x2 .le. i1 + 1)  rslts(33) = 1
	if (i1 .ne. c12 - (0.0, 2.0)) rslts(34) = 1
	if (d1 .gt. -x2)     rslts(35) = 1
	if (d1 .lt. 0)       rslts(36) = 1

c --- check results:

	call check(lrslts, expect, n)

	data expect / TRUE, TRUE, 0, TRUE, 0, 0,
     +                TRUE, 0, TRUE, 0,
     +                TRUE, 0, 0, TRUE,
     +                TRUE, 0, 0, TRUE,
     +                0, TRUE, TRUE, 0,
     +                TRUE, 0, 0, TRUE,
     +                1, 0, 1, 0, 1, 0,
     +                1, 0, 1, 0       /
	end


	real function xfn5(c)
	logical c
	xfn5 = 5
	if (c) xfn5 = - xfn5
	end
