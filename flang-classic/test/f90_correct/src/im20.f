** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Character statement functions, and statement function
*   arguments of type character.

	program p
	parameter(n=21)
	character*4 rslts(n), expect(n)
        integer*4 iir(n), iie(n)
        equivalence (iir, rslts)
        equivalence (iie, expect)
	logical lrslts(n), lexpect(2)
	integer irslts(n), iexpect(3)
	character*8 rslts17_18
	equivalence (irslts, lrslts, rslts), (rslts17_18, rslts(17))
	equivalence (expect(19), iexpect), (expect(15), lexpect)

	character c2*2, c2x*2, c10*10, c1*1
	character cx*1, cext*8, cab*2
	character first*1

	integer ilen
	external ilen

c --- declare the statement functions:

	character ca2*2, ca3*3, ca4*4, ca1*1, cb3*3, ce3*3,
     +            cc1*1, cd3*3, cc3*3, ca8*8
	integer lp1
	logical ge

c --- define the statement functions:

	ca2(c2) = c2
	ca3(i, j) = cext(i:j)
	ca4(c2) = 'a' // c2 // 'd'
	ca1(i) = char(i + 1)
	cb3(i, c2) = c2 // char(i)
	ce3() = 'xyz'
	cc1(c2) = c2
	cd3(c2) = c2
	lp1(c10) = ilen(c10) + 1
	cc3(c2) = first(c2) // c2
	ge(c2, c2x) = c2 .ge. c2x
	ca8(c1) = ca4(c1 // 'z') // 'r' // ca4(c1)

	data i1, i2 / 1, 2 /, cx, cext, cab / 'x', 'external', 'ab' /

c  --- reference the statement functions:

c --- tests 1 - 7:

	data (expect(i), i = 1, 7) /
     +     'ab  ', 'xd  ', 'xted', '.ex ', 'axtd', '\010   ', 'ly  '/

	rslts(1) = ca2('ab')
	rslts(2) = ca2(cx // 'def')
	rslts(3) = ca3(2, 4) // 'def'
	rslts(4) = '.' // ca3(i1, i2) // '.'
	rslts(5) = ca4(cext(i2:i2+1))
	rslts(6) = ca1(7)
	rslts(7) = ca1(ichar('k')) // 'y'

c --- tests 8 - 14:

	data (expect(i), i = 8, 14) /
     +        'ab\07 ', 'xyz ', 'a   ', 'ab  ', 'rrn ', 'xx  ', 'dde.'/

	call cassign(rslts(8), cb3(7, 'ab'))
	call cassign(rslts(9), ce3())
	call cassign(rslts(10), cc1('abc'))
	call cassign(rslts(11), cd3('abcde'))
	call cassign(rslts(12), cc3(cext(5:6)))

	rslts(13) = cc3(cx)
	rslts(14) = cc3('de') // '.'

c --- tests 15 - 18:

	data lexpect / .true., .false. /

	lrslts(15) = ge('ab', 'ab')
	lrslts(16) = ge(cx, 'y')

	data (expect(i), i = 17, 18) / 'axzd', 'rax ' /

	rslts17_18 = ca8(cx)

c --- tests 19 - 21:

	data iexpect / 2, 11, 4 /

	irslts(19) = lp1('a')
	irslts(20) = lp1('abcdefghijkl')
	irslts(21) = lp1('abc   ')

c  --- check the results:

	call check(iir, iie, n)
	end

c --------------------

	function ilen(c)
	character*(*) c

	do 10 i = 1, len(c)
10          if (c(i:i) .eq. ' ')  goto 20
20	ilen = i - 1
	end

c --------------------

	character*1 function first(c)
	character*2 c
	first = c
	end

c --------------------

	subroutine cassign(a, v)
	character*4 a, v*(*)
	a = v
	end
