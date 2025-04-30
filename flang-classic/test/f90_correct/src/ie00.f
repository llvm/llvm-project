** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   ENTRY statements in subroutines.

	parameter(n = 24)
	integer rslts, expect(n)
	integer TRUE
	parameter (TRUE = -1)
	common rslts(50)

	character*2 cat, cxb

c --- tests 1 - 2:

	data expect(1), expect(2) / 13, 11 /

	call e2
	call sub1()
	call e1()

c --- tests 3 - 5:

	data expect(3), expect(4), expect(5) / 8, 8, 1 /

	call sub2( rslts(3) )
	call e3( rslts(4) )
	j = 0
	call e4(j)
	rslts(5) = j

c --- tests 6 - 10:

	data (expect(i), i = 6, 10) / 7, 6, -5, 5, -3 /

	call sub3(j, 2, 5)
	rslts(6) = j
	call e5(rslts(7), 4, 10)
	call e6(rslts(8), 3, 8)
	call e7(rslts(9), 3, 8)
	call e8(j, -1, 3)
	rslts(10) = j

c --- tests 11 - 16:

	data (expect(i), i = 11, 16) / 7, -1, 17, 5, 34, 1 /

	call sub4( rslts(11) )
	rslts(12) = -2
	call e4a
	call e4b(rslts(13), 9)
	call e4c(3.91)
	call e4d(3.1, rslts(15), 13, 3)
	j = 0
	call e4e(j)
	rslts(16) = j

c --- tests 17 - 24:

	data (expect(i), i = 17, 24) 
     +       / '171'o, TRUE, '142'o, 8, 6, 2, 1, 4 /

	cxb = 'xb'
	call e5a(cxb)
	rslts(17) = ichar(cxb(1:1))

	cat = 'at'
	call e5b(cat, rslts(18))
	rslts(19) = ichar(cat(1:1))

	cat(1:1) = char(7)
	call e5c(10, cat)
	rslts(20) = ichar(cat(1:1))

	call e5d('abcdef', 'ab', 'x', rslts(21), rslts(22), rslts(23))
	call e5e(rslts(24), cat // cat)

c --- check results:

	call check(rslts, expect, n)
	end

ccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine sub1
	common i(50)
	data i(1), i(2) / 2 * 10 /

	i(2) = i(2) + 1

	entry e1
	entry e2

	i(1) = i(1) + 1
	end

ccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine sub2(i)
	entry e3(i)
	i = 7
	entry e4(i)
	i = i + 1
	end

ccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine sub3(i, j, k)
	i = j + k
	return

	entry e5(i, k, j)
	i = j - k
	return

	entry e6(k, i, j)
	entry e7(k, j, i)
	k = i - j
	return

	entry e8(j, k, i)
	j = i * k
	end

ccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine sub4(i)
	common r(50)
	integer r
	common /l/ locvar
	real j

	data locvar / 7 /

	i = locvar
	locvar = locvar + 1
	return

	entry e4a
	r(12) = r(12) + 1
	return

	entry e4b(i, ival)
	i = locvar + ival
	return

	entry e4c(j)
	r(14) = j + 1.1
	return

	entry e4d(x, i, k, ival)
	i = int(x) + k + 3*ival + locvar

	entry e4e(i)
	i = i + 1
	end

cccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine sub5()
	entry e5a(c)
	implicit character*1 (c)
	logical l
	parameter(i5 = 5)
	character*2 c, cc*(*), dd*(*)

	cincr(cx) = char(ichar(cx) + 1)

	entry e5b(c, l)
	if (c(1:1) .eq. 'a') then
		l = c(2:2) .eq. 't'
	endif
10	c(1:1) = cincr(c(1:1))
	return

	entry e5c(i, c)
	goto 10

	entry e5d(cc, c, dd, i, j, k)
	i = len(cc)
	j = len(c)
20	k = len(dd)
	return

	entry e5e(k, dd)
	goto 20
	end
