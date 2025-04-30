** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   KANJI - string comparison operators, nindex instrinsic.

	program p
	parameter(N=14)

	ncharacter a*3, b*2, c*5, d*3
	common a, b
	data a / nc'abc'/
	data b / nc'ab' /
	data c / nc'abc  ' /
	data d / nc'abd'/

	integer expect(N), rslts(N), T, F
	parameter (T = -1, F = 0)

	data expect /	F, T, T, T, T, F, T,
     +			1, 2, 2, 0, 5, 1, 0/

	! -------- test comparisons:

	rslts(1) = a .eq. c		!	F
	rslts(2) = b .gt. a		!	T
	rslts(3) = d .ge. nc'abc'	!	T
	rslts(4) = nc'ab' .ne. c	!	T
	rslts(5) = llt(a, d)		!	T
	rslts(6) = lle(b, nc'ab    \0')	!	F
	rslts(7) = lgt(nc'z', c)	!	T

	! --------- test NINDEX:

	rslts(8) = index(c, nc'a')		! 1
	rslts(9) = nindex(c, nc'b')		! 2
	rslts(10) = index(nc'aabcde', a)	! 2
	rslts(11) = nindex(a//b, nc'Z')		! 0
	rslts(12) = index(c//c, nc' a')		! 5
	rslts(13) = nindex(a(2:3), nc'bc')	! 1
	rslts(14) = index('a', 'acd')		! 0

	call check(rslts, expect, N)
	end
