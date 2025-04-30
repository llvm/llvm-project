** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   %LOC intrinsic.

	program p
	common rslts
	integer rslts(16), expect(16)

	equivalence (k, kk)
	complex a(2:4, 2:3)

	common /c/ c1, c2, c3, c4
	character c1*1, c2*3, c3(5), c4(2)*3

	integer iarr(2)
	equivalence (i,iarr(2))

c    tests 1 - 5:  LOC of local and common variables, arrays and scalars:

	rslts(1) = %loc(rslts) - %loc(rslts(3))
	j = %loc(rslts(3))
	data i3 / 3/
	rslts(2) = j - %loc( rslts(i3-1) )
	if (%loc(i) .gt. %loc(iarr))  rslts(3) = 3
	rslts(4) = %loc(k) - %loc(kk)
	rslts(5) = %loc(a(4, i3-1)) - %loc(a)

c    tests 6 - 8:  LOC of character variables:

	rslts(6) = %loc(c3) - %loc(c1)
	rslts(7) = (%loc(c2(i3:i3)) + 1) - %loc(c2(2:))
	rslts(8) = %loc(c4(i3-2)(:)) - %loc(c4(2)(i3-2:3))

c    tests 9 - 13:  LOC as argument, LOC of dummies:

	rslts(9) = jf(%LOC(rslts(10)), rslts(10), %val(%loc(rslts(13))) )

c    tests 14 - 16:  referencing memory thru base array:

	call sub

c    check results:

	call check(rslts, expect, 16)

	data expect / -8, 4, 3, 0, 16,
     +                4, 2, -3,
     +                77, 4, 0, 0, 66,
     +                8, 22, 33       /
	end


	integer function jf(ar10, r10, r13)
	common rslts
	integer rslts(16), r13, ar10
	data i0 / 0/

	rslts(10) = %loc(rslts(11)) - ar10
	rslts(11) = %loc(r10) - ar10
	rslts(12) = %loc(jf) * i0
	r13 = 66

	jf = 77
	end


	subroutine sub
	common /kkkk/k
	integer rslts(16), a(1), kk
	common rslts
	save a, kk
	data i15 / 15 /

	j = (%loc(rslts) - %loc(a)) / 4
	jj = (%loc(a) - %loc(kk)) / 4
c       assign to rslts(14) via array a:
	a(14 + j) = 8

c       ... break basic block.
!        if (i15 .gt. 0)  k = 1
	if (ifff(k) .gt. 0) k = 1

c       assign to rslts(15) via a:
	a(j + i15) = 22
c       assign to scalar kk via a:
	a(1-jj) = 33
c ... break basic block since scheduler will  not recognize
c     dependency between a(1-jj) and kk.
!        if (i15 .gt. 0)  k = 1
	if (ifff(k) .gt. 0) k = 1
	rslts(16) = kk
	end
	function ifff(ii)
	ifff = ii+1
	end
