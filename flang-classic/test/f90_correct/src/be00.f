** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   EQUIVALENCE statements - local variables and arrays.

	program equivalence_statements
	parameter(N = 15)
	integer rslts(N), expect(N)

c  --- tests 1 - 6:

c     test multiple items in equivalence group:
	integer a(4), b(4), c(4), d(4)
	equivalence (a(1), b(2), c(3), d(4))

c     now test reverse order:
	integer*2 aa(4), bb(2), cc(3), dd(4)
	equivalence (dd(4), cc(3), bb(2), aa(1))

c  --- tests 7 - 8:

c     test redundant equivalences and constant expressions for
c     subscripts:
	real x(4), y(4), z(4)
	equivalence (x(2*2), y(-(-2)), y(4-2)),
     +              (z(10/5), y(7/2)), 
     +              (x(1 + 3), z((((1))))),
     +              (x(3), y(1)),
     +              (y(4), z(3))

c  --- tests 9 - 15:

c     test equivalences of character arrays, substrings, etc.
c     storage is laid out as follows:
c                          +-----+
c    c1:                   |     |
c                          +-----+-----------------+
c    d4:                   |                       |
c                    +-----+-----+-----------+-----+-----+
c    e2(3):          |           |           |           |
c              +-----+-----+-----+-----+-----+-----+-----+
c    f1(3,2):  | 1,1 | 2,1 | 3,1 | 1,2 | 2,2 | 3,2 |
c              +-----+-----+-----+-----+-----+-----+-----+
c    g1(-1:2):                   |     |     |     |     |
c        +-----+-----+-----+-----+-----+-----+-----+-----+
c    h8: |  a  |  b  |  c  |  d  |  e  |  f  |  g  |  h  |
c        +-----+-----+-----+-----+-----+-----+-----+-----+
c  i3(99998:99999):  |                 |                 |
c                    +-----------------+-----------------+

	character c1*1, d4*4, e2(3)*2, f1(3,2)*1,
     +            g1(-1:2)*1, h8*8, i3(99998:99999)*3

	equivalence (c1, d4) ,
     +              (e2(3), d4(4:4)),
     +              (f1(3, 1), e2(1)(2:))
	equivalence (g1(-1)(:), d4(1 + 1: 3)),
     +              (h8(5:), f1(1, 2)),
     +              (g1, i3(99998)(3:) )

C  ---------------------------------------------------------------
C  ---------------------------------------------------------------
C  --- tests 1 - 6:

	data i4 / 4 /
	d(i4) = 4
	rslts(1) = a(1)
	rslts(2) = b(2)
	rslts(3) = c(3)

	aa(1) = 2
	rslts(4) = dd(4)
	rslts(5) = cc(i4 - 1)
	rslts(6) = bb(2)

C  --- tests 7 - 8:

	x(4) = 5.1
	rslts(7) = y(2)
	rslts(8) = z(i4 - 3)

C  --- tests 9 - 15:

	data h8 / 'abcdefgh' /
	rslts(9 ) = ichar( c1 )
	rslts(10) = ichar( d4(1:1) )
	rslts(11) = ichar( e2(3)(2:1+1) )
	rslts(12) = ichar( f1(3, 2) )
	rslts(13) = ichar( g1(-1) )
	rslts(14) = ichar( h8(:1) )
	rslts(15) = ichar( i3(99999)(3: ) )

C  --- check results:

	call check(rslts, expect, N)
	data  expect/ 4, 4, 4, 2, 2, 2,
     +                5, 5,
     +                100, 100, 104, 103, 101,  97, 104  /
c                     'd', 'd', 'h', 'g', 'e', 'a', 'h'
	end
