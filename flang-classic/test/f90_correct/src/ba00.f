** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   DIMENSION, COMMON, and type specification statements.

C   NONSTANDARD:
C     Use of size specification on integer, etc. (VMS)
C     %eject directives (VMS)

	program p
c    tests 1 - 3:   DIMENSION Statements:
	call s1

c    tests 4 - 18:  COMMON Statements:
	call s2

c    tests 19 - 37: Type Specification Statements:
	call s3

c    define expected results array and call check:
	call s4
	end

c--------------------------------------------------------------
c
c    tests 1 - 3: DIMENSION Statements:
c      (1) DIMENSION statements which precede and follow corresponding
c          type specification statement.
c      (2) One and more than one array specified in DIMENSION stmt.
c      (3) Array with seven dimensions.
c      (4) redefinition of an intrinsic name to be an array.

	subroutine s1
	integer rslts
	dimension rslts(37)
	common /r/ rslts
	common expect
	integer*2 a
	dimension expect(37), a(3,2), int(2,2,2,2,2,2,2)
	integer expect
	data i2 / 2/

	rslts(1) = 2
	expect(1) = 2
	a(3,2) = 3
	rslts(2) = a(3, i2)
	int(1,1,1,2,1,1,1) = 7
	rslts(3) = int(1, i2-1, 1, i2, 1, 1, 1)

	return
	end
%eject
c--------------------------------------------------------------
c
c    tests 4 - 18: COMMON Statements:
c      (1) Dimensions of array common block element specified before
c          within, and following appearance of array name in COMMON.
c      (2) Blank common different sizes in different subprograms.
c      (3) Blank common referenced by omitting cblock name, and by "//".
c      (4) Multiple COMMON statements for a given common block.
c      (5) Multiple common blocks defined by a single COMMON statement.
c      (6) Common block name same as variable names.
c      (7) 31 character common block name with special chars ($, _).
C 			modified for HPF -- lfm
c      (8) common block member (ff) unreferenced.
c      (9) 1 byte common block.

	subroutine s2
	integer b(2)
	common //expect//r
	common /s2com/a /s2com/b, /s2com/c, d /s2com/e,ff,g,/s2com/h
	common /r/ rslts(37)
	integer rslts, expect(37), a, b, c, d, e, g, h, r,
     +          s, t, u, v, w, x
	common s
	common //t//u,v//w,//x,/c_$4567890123423456789/ f
	logical * 1 f

c   -- assign values to a thru h and r thru x:
	call s2x

	rslts(4) = a
	rslts(5) = b(2)
	rslts(6) = c
	rslts(7) = d
	rslts(8) = e
	rslts(9) = and(1, f)
	rslts(10) = g
	rslts(11) = h

	rslts(12) = r
	rslts(13) = s
	rslts(14) = t
	rslts(15) = u
	rslts(16) = v
	rslts(17) = w
	rslts(18) = x

	return
	end

	subroutine s2x
	common // expect(37), ir(7)
	common /s2com/ ia(9)
	common /c_$4567890123423456789/ if
	logical*1 if

	do 10 i = 1, 7
10		ir(i) = 2 * i - 3

	do 11 i = 1, 9
11		ia(i) = i * 10

	if = .true.
	end
%eject
c--------------------------------------------------------------
c
c    tests 19 - 37:  Type specification statements:
c      (1) All forms of CHARACTER statements including, constant
c          expression for length spec, default length of 1,
c          over-riding length spec, optional comma after length.
c      (2) All allowed type specification statements, including
c          length specifiers for INTEGER, REAL, LOGICAL, and COMPLEX.
c      (3) Redundant specification of intrinsic type.
c      (4) Specification of type for generic name (which should
c          not remove generic property of identifier).

	subroutine s3
	common /r/ rslts(37)
	integer rslts

	character x1
	character*3, x2
	character x3, x4
	character x5*2
	character*4 x6
	character*(((3+7))) x7
	parameter(mm = 3)
	character*(mm*4), x8
	character*2 x9*(9-mm), x10*5, x11, x12*(1), x13*(mm)

	integer i
	integer*4 j
	integer*2 k
	logical l
	logical*4 m
	logical*1 n
	real o
	real*4 p
	real*8 q
	doubleprecision r
	complex s
	complex*8 t

	integer iabs
	character*7 max

	rslts(19) = len(x1)
	rslts(20) = len(x2)
	rslts(21) = len(x3)
	rslts(22) = len(x4)
	rslts(23) = len(x5)
	rslts(24) = len(x6)
	rslts(25) = len(x7)
	rslts(26) = len(x8)
	rslts(27) = len(x9)
	rslts(28) = len(x10)
	rslts(29) = len(x11)
	rslts(30) = len(x12) * len(x13)

	data i, j, k / 2, 3, 4/
	rslts(31) = i + j + k

	data l, m, n / 3 * .true. /
	rslts(32) = AND(1, l .and. m .and. n)

	data o, p / 5.4, 2.61 /
	rslts(33) = o + p

	data q, r / 123456789012.1D0, 123456789010.0D0 /
	rslts(34) = q - r

	data s, t / (0.0, 1.0), (2.0, 3.1) /
	rslts(35) = aimag(s) + aimag(t)

	rslts(36) = iabs(-3)
	rslts(37) = max(3, 8)

	return
	end
%eject
c--------------------------------------------------------------
c
c    Define expected results array and call check:

	subroutine s4
	common /r/ rslts(37)
	integer rslts
	common expect(37), ifillit(7)
	integer expect
	parameter (x = 0)

c    ---- tests 1 - 3:
	data expect / x, 3, 7,
c    ---- tests 4 - 11:
     +                10, 30, 40, 50, 60, 1, 80, 90,
c    ---- tests 12 - 18:
     +                -1, 1, 3, 5, 7, 9, 11,
c    ---- tests 19 - 30:
     +                1, 3, 1, 1, 2, 4, 10, 12, 6, 5, 2, 3, 
c    ---- tests 31 - 37:
     +                9, 1, 8, 2, 4, 3, 8          /

	call check(rslts, expect, 37)
	end
