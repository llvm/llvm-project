** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Relational operators (EQ - LT) for numeric types, 
*   character, and logical, used outside of IF conditions.
*   Constant folding and expressions
*   requiring type conv. of one operand are not tested.

	parameter (N = 36)
	implicit logical(F, l-W), double precision (d), complex (c)
	integer TRUE
	parameter (TRUE = -1)
	dimension rslts(N)
	integer expect(N)
	character cab*2, cabx*3, ca*1, cb*1, cabb*3, cabc*3,
     +            cabdd*4, cabcdxx*6, cabcdef*6, cabcdefy*7

	data expect / TRUE, 0, TRUE, 0, TRUE, 0, TRUE, TRUE,
     +                0, TRUE, 0, 0, TRUE, TRUE,
     +                0, TRUE, 0, 0, 0, 0,
     +                TRUE, 0, 0, TRUE,
     +                0, TRUE, 0, TRUE, 0, 0, TRUE, TRUE,
     +                TRUE, TRUE, 0, 0               /

C  --- tests 1 - 8:     integer comparisons.

	data i2, i3, i4, in3, in1, i0, i99999, i1/
     +        2,  3,  4,  -3,  -1,  0,  99999,  1/

	rslts(1) = i2 .eq. i1 + 1
	rslts(2) = i3 .gt. i4
	rslts(3) = i2 + i2 .ge. i3
	rslts(4) = ior(i4, 0) .le. or(i3, 0)
	rslts(5) = 3 .ne. in3
	rslts(6) = i3 .lt. in3
	rslts(7) = in1.lt.i0
	rslts(8) = i99999.ge.99999

C  --- tests 9 - 14:    real comparisons.

	data x2, x201, x31, x3, x523, xn25, x25,  x0, xnp1/
     +      2.0, 2.01, 3.1,3.0, 5.23, -2.5, 2.5, 0.0, -.1   /

	rslts(9) = x3 - 1.0 .eq. x201
	rslts(10) = x31 .ge. x3 + x0
	rslts(11) = x523 .lt. 5.23
	rslts(12) = xn25 .ne. xn25
	rslts(13) = 2.5 .gt. x0
	rslts(14) = xnp1. le .x0

C  --- tests 15 - 20:   double precision comparisons.

	data d50,  d5,   d25,  dp25,  d0,   dn2111,  dn25 /
     +       5d1, 5d0, 2.5d0, 25d-2, 0d0, -2.1d-11, -2.5d0 /

	rslts(15) = d50 .ne. d50
	rslts(16) = d5 + d0 .lt. d50
	rslts(17) = d25 .eq. dp25
	rslts(18) = -1.0D0 .ge. d0 * 2
	rslts(19) = dn2111 .gt. -2.1D-11
	rslts(20) = d25 .le. - d25

C  --- tests 21 - 24:   complex comparisons (only EQ, NE allowed).

	data c10, c11 / (1.0, 0.0), (1.0, 1.0) /

        rslts(21) = c11 .eq. c11
        rslts(22) = (1.0, 1.0) .eq. c10
        rslts(23) = c10 .ne. (1.0, 0.0)
        rslts(24) = c10 .ne. c11

C  --- tests 25 - 32:   character comparisons.

	data cab,  cabx,  ca,  cb,  cabb,  cabc,  cabdd,  cabcdxx /
     +       'ab', 'ab ', 'a', 'b', 'abb', 'abc', 'abdd', 'abcd  ' /
	data cabcdef, cabcdefy / 'abcdef', 'abcdef!' /

	rslts(25) = cab .ne. cab
	rslts(26) = cab .eq. cabx
	rslts(27) = ca.gt.cb
	rslts(28) = cabx .le. cab
	rslts(29) = cabb .ge. 'abc'
	rslts(30) = 'ab' .lt. ca
	rslts(31) = cabdd .gt. cabcdxx
	rslts(32) = cabcdef .lt. cabcdefy

C  --- tests 33 - 36:   logical comparisons, EQ and NE (extension to
C                       f77 standard).

	data t, f / .true., .false. /

	rslts(33) = t .eq. t
	rslts(34) = (t .or. t) .ne. f
	rslts(35) = f .eq. .true.
	rslts(36) = .false. .ne. (f .and. t)

C  --- check results:

	call check(rslts, expect, N)
	end
