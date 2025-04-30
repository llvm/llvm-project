** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   KANJI - functions, entries, dummy args of type NCHARACTER.

	program p
	parameter(N=14)

	ncharacter*1 a1
	ncharacter*2 a2
	ncharacter*3 a(2,2)
	ncharacter*4 a4
	ncharacter*5 a5(6)
	common a2, a5

	! declare nchar functions:
	ncharacter nextc*1, alpha*3, char1*2, char2*2

	integer expect(N), rslts(N)

	data i1 / 1 /
	data expect /	66, 67,
     +			97, 99, 'a1a1'x, 97,
     +			98, 99, 67, 66,
     +			3, 4, 4,   90   /

	! -------- test function nextc:

	a1 = nextc(nc'A')
	rslts(1) = ichar(a1)		!	'B'
	rslts(2) = ichar(nextc(a1))	!	'C'

	! --------- test function alpha:

	a4 = alpha()
	rslts(3) = ichar( a4(1:1) )	!	'a'
	rslts(4) = ichar( a4(3:3) )	!	'c'
	rslts(5) = ichar( a4(4:4) )	!	' '
	a4 = alpha() // alpha()
	rslts(6) = ichar( a4(4:4) )	!	'a'

	! --------- test function char1 and entry char2:

	a2 = char1(nc'abcdefg')
	rslts(7) = ichar( a2(1:1) )	!	'b'
	rslts(8) = ichar( a2(2:2) )	!	'c'

	a2 = nc'BC'
	a2 = char2(a2//a2//a2)	! 'BCBCBC'(4:5)
	rslts(9) = ichar( a2(1:1) )	!	'C'
	rslts(10) = ichar( a2(2:2) )	!	'B'

	! ---------- test function ifunc:

	rslts(11) = ifunc( NC'abc' )	!	3
	rslts(12) = ifunc( a2 // NC'xx')!	4
	rslts(13) = ifunc( a5(i1)(2-i1:3+i1) ) ! 4

	! ---------- test function jfunc:

	a(2,2) = nc'XYZ'
	rslts(14) = jfunc(a, 2)		!	'Z'
	
	call check(rslts, expect, N)
	end

	ncharacter function nextc(c)
	ncharacter*1 c
	i = ichar(c) + 1
	nextc = nchar(i)
	end

	ncharacter*(*) function alpha
	alpha = nc'abcdefghijklmnopqrstuvw'
	return
	end

	function char1(c)
	ncharacter*2 char1
	ncharacter*(2) char2
	ncharacter*200 c
	char1 = c(2:3)
	return

	entry char2(c)
	char2 = c(4:5)
	end

	function ifunc(c)
	ncharacter*(*) c
	ifunc = len(c)
	end

	function jfunc(c, n)
	ncharacter c(n,n)*(*)
	jfunc = ichar( c(2,2)(3:3) )
	end
