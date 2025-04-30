** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Constant folding of relational operators (EQ, NE, GT, GE,
*   LE, LT) for numeric types, character, and logical.  
*   Constant expressions requiring type conversion of one of
*   the arguments are not tested.

	parameter (N = 36)
	implicit logical(F, l-W)
	dimension rslts(N)
	integer expect(N)

	parameter(p1 = 2 .eq.2, p2 = 3 .gt. 4)
	parameter(p3 = 2.0 .eq. 2.01, p4 = 3.1.ge.3.0)
	parameter(p5 = 5.0D1.ne.5.0D1, p6=5.0D0.lt.5.0d1)
	parameter(p7=(1.0,1.0).eq.(1.0,1.0), p8=(1.0,1.0).eq.(1.0,0.0))
	parameter(p9 = 'ab' .ne. 'ab', p10 = 'ab' .eq. 'ab ')
	parameter(p11 = .true..eq..true., p12 = .true..ne..false.)
	parameter(TRUE = -1)

	data expect / TRUE, 0, TRUE, 0, TRUE, 0, TRUE, TRUE,
     +                0, TRUE, 0, 0, TRUE, TRUE,
     +                0, TRUE, 0, 0, 0, 0,
     +                TRUE, 0, 0, TRUE,
     +                0, TRUE, 0, TRUE, 0, 0, TRUE, TRUE,
     +                TRUE, TRUE, 0, 0               /

C  --- tests 1 - 8:     integer comparisons.

	rslts(1) = p1
	rslts(2) = p2
	rslts(3) = 4 .ge. 3
	rslts(4) = '4'x .le. '3'x
	rslts(5) = 3 .ne. -3
	rslts(6) = 3 .lt. -3
	rslts(7) = -1.lt.0
	rslts(8) = 99999.ge.99999

C  --- tests 9 - 14:    real comparisons.

	rslts(9) = p3
	rslts(10) = p4
	rslts(11) = 5.23 .lt. 5.23
	rslts(12) = -2.5 .ne. -2.5
	rslts(13) = 2.5 .gt. 0.0
	rslts(14) = -.1. le .0.0

C  --- tests 15 - 20:   double precision comparisons.

	rslts(15) = p5
	rslts(16) = p6
	rslts(17) = 2.5D0 .eq. 2.5D-1
	rslts(18) = -1.0D0 .ge. 0.0D0
	rslts(19) = -2.1D-11.gt.-2.1D-11
	rslts(20) = 2.5d0 .le. -2.5d0

C  --- tests 21 - 24:   complex comparisons (only EQ, NE allowed).

	rslts(21) = p7
	rslts(22) = p8
	rslts(23) = (1.0, 0.0) .ne. (1.0, 0.0)
	rslts(24) = (1.0, 0.0) .ne. (1.0, 1.0)

C  --- tests 25 - 32:   character comparisons.

	rslts(25) = p9
	rslts(26) = p10
	rslts(27) = 'a'.gt.'b'
	rslts(28) = 'ab ' .le. 'ab'
	rslts(29) = 'abb' .ge. 'abc'
	rslts(30) = 'ab' .lt. 'a'
	rslts(31) = 'abdd' .gt. 'abcd  '
	rslts(32) = 'abcdef' .lt. 'abcdef!'

C  --- tests 33 - 36:   logical comparisons, EQ and NE (extension to
C                       f77 standard).

	rslts(33) = p11
	rslts(34) = p12
	rslts(35) = .false. .eq. .true.
	rslts(36) = .false. .ne. .false.

C  --- check results:

	call check(rslts, expect, N)
	end
