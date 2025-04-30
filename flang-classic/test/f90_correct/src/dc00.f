** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   INDEX, LGE, LGT, LLE, and LLT intrinsics, and character
*   comparison operations - .EQ., etc.

	program p
	character*15 c2
	data c2 / 'aababcabcdabcde' /

	call dotest(c2)
	end

	subroutine dotest(c2)
	character*(*) c2

	parameter (n = 22)
	integer rslts(n), expect(n)
	logical lrslts(n)
	equivalence (lrslts, rslts(1))

	character c1*13, c3*4

	data c1, c3 / 'abcdeabcdabcx', 'abcd' /
	data i2, i6 / 2, 6 /

c -------- tests 1 - 6: INDEX intrinsic:

	rslts(1) = index('abc', 'a') + index('abc', 'a')
	rslts(2) = index('abc', 'c') + 10 * index('abc', 'd')
	rslts(3) = index(c1, 'cde') + index('d', 'd  ')
	rslts(4) = index(c2, c3)
	rslts(5) = index(c2(:14), 'e')
	rslts(6) = index(c1, c3(3:) )

c -------- tests 7 - 14: LGE - LLT intrinsics:

	lrslts(7) = lge('abcd', c3)
	lrslts(8) = lgt(c3, 'abce')
	lrslts(9) = lle('bbc', 'abc')
	lrslts(10) = llt(c3, c3(:)) .neqv. .true.
	lrslts(11) = lge(char(6), '  ')
	lrslts(12) = lgt('abcd', 'abc')
	lrslts(13) = lle(c3 // 'd', c1(1:i6) )
        rslts(14) =  0
	if ( llt('ab  A', c2(2:3)) )   rslts(14) = 2

c -------- tests 15 - 22: character comparisons:

	lrslts(15) = c3 .eq. c3 // '    '
	lrslts(16) = c2(i2:15) .ne. c2(2:)
	lrslts(17) = 'b' .ge. 'a'
	lrslts(18) = char('141'o) .gt. c2(1:1)
	lrslts(19) = c3(:) .le. c2(7:10)

	data (rslts(i6), i6 = 20, 22) / 3 * 0 /
	if (c3 .lt. c1)  rslts(20) = 8
	if (c3 .eq. c1)  rslts(21) = 1
	if (c1 .ne. c2)  rslts(22) = 1

c --------- check results:

	call check(rslts, expect, n)
	data expect / 2, 3, 3, 7, 0, 3, 
     +                -1, 0, 0, -1, 0, -1, -1, 0,
     +                -1, 0, -1, 0, -1, 8, 0, 1  /
	end
