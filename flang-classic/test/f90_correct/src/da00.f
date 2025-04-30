** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   LEN (character length) intrinsic.

C   NONSTANDARD:
C      Indeterminate length concats as operand to LEN.

	program p
	parameter ( n = 30 )
	integer rslts(n), expect(n)
	common /r/ rslts

	character a*1, b*2, c*300, d(3:5)*10
	common a, d
	character *11, f, sub*200, func2*12

	data  i2, i3, i10 / 2, 3, 10 /

c ------ tests 1 - 5:   LEN of constant expression, variable, array element:

	rslts(1) = len('a')
	rslts(2) = len(('ab' // ('dc')))
	rslts(3) = 4 * len(a)
	rslts(4) = len(b) - 4
	rslts(5) = len ( d(4) )

c ------ tests 6 - 10:  LEN of substring expression:

	rslts(6) = len(d(i)(1:1))
	rslts(7) = len( b(:) )
	rslts(8) = len( c(i10: i10+3) )
	rslts(9) = len( c(i10*i2 : i3*i10+i2) )
	rslts(10) = len( d(i2)(5: i10-2) )

c ------ tests 11 - 15: LEN of concatenation expression:

	rslts(11) = len( a // b)
	rslts(12) = len(a//a//b(2:) )
	rslts(13) = len(d(i) // a // 'dc')
	rslts(14) = len( c(i2:i10) // a)
	rslts(15) = len(b(1:1) // c(i3:i10-i2) // c(i3:i3) )

c ------ tests 16 - 20: LEN of CHAR intrinsic and function call:

	rslts(16) = len( char(i10+i3) )
	rslts(17) = len('a' // char(7) )
c    --- function f will be called, although not necessary:
	rslts(20) = 0
	rslts(18) = len( f(i10) )
	rslts(19) = len(b // f(i10) // c)

c ------ tests 21 - 30: LEN of expressions involving passed length
c                       character dummy arguments:

	c = sub(a, b, c, i, d, d(4) )
        c = func2(i, j)

c ------ check results:

	call check(rslts, expect, n)
	data expect / 1, 4, 4, -2, 10,
     +                1, 2, 4, 13, 4,
     +                3, 3, 13, 10, 8,
     +                1, 2, 11, 313, 2,
     +                3, 2, 300, 10, 10,
     +                3, 12, 320, 400, 14  /
	end

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

	character*(*) function sub(a, b, c, i, d, e)
	character*2 b, a*3, c*(*), d*(*), e*(*)

	common /r/ rslts(30)
	integer rslts

	rslts(21) = len(a)
	rslts(22) = len(b)
	rslts(23) = len(c)
	rslts(24) = len(d)
	rslts(25) = len(e)

	rslts(26) = len( c(298:) )
	rslts(27) = len( e // 'ab' )
	rslts(28) = len( c // d // e )
	rslts(29) = len( sub ) + len(sub(2:) // 'a')

	end

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

	character*11 function f(i)
c  --- this function should never be called.
	common /r/ rslts(30)
	integer rslts
	rslts(20) = rslts(20) + 1
	end

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

	character*(14) function func2(i, j)
	common /r/ rslts(30)
	integer rslts

	rslts(30) = len(func2)
	end
