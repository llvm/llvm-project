** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Hollerith constants.

C   NONSTANDARD:
C     Hollerith constants.
C     'or' intrinsic (non-VMS).

	program p
	implicit logical (l), complex (c)

	parameter(n = 19)
	integer rslts(n), expect(n)
	integer x

	data i, i1, x, l / 4h((((, 1, 3h///, 1Hc /

	rslts(1) = i
	rslts(2) = or(x, 0)
	rslts(3) = or(0, l)

	i = 1Hc
	x = 2H
	l = 4h,,{,
	rslts(4) = i
	rslts(5) = or(x, 0)
	rslts(6) = or(l, 0)

	rslts(7) = i1 * 4habcd
	rslts(8) = 2heg + i1

	if (i .eq. 1hc)  rslts(9) = 1
	if (4hc    .eq. i)  rslts(10) = 1
	rslts(11) = 1
	if (2hc  .eq. i)    rslts(11) = 0
	rslts(12) = 0
	if (x .ne. 2h  )    rslts(12) = 1

	call assign(1hx, rslts(13))
	call assign(4h)=,', rslts(14))
	rslts(15) = or(3h2e3, 0)

	c = ccopy(8hdefghijk)
	call assign( real(c), rslts(16))
	call assign( aimag(c), rslts(17))

	call acopy(12hABCDEFGHIJKL, rslts(18), rslts(19) )

c --- check results:

	call check(rslts, expect, n)

	data expect / '((((', '/// ', 'c   ',
     +                'c   ', '    ', ',,{,',
c          tests 7 - 15:
c  BIG ENDIAN
c     +                'abcd', 'eg !', 
c  LITTLE ENDIAN
     +                'abcd', 'fg  ', 
     +                1, 1, 0, 0, 
     +                'x   ', ')=,''', '2e3 ',
c          tests 16 - 19:
     +                'defg', 'hijk', 'ABCD', 'IJKL'  /
	end

cccccccccccccccccccccccccccccccccccccccccccc

	subroutine assign(i, j)
	j = i
	end

	complex function ccopy(c)
	complex c
	ccopy = c
	end

	subroutine acopy(a, i, j)
	integer a(3)
	i = a(1)
	j = a(3)
	end
