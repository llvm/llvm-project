** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   KANJI - NCHARACTER literal constants, constant folding, PARAMETER stmt.

	parameter (N=17)
	
	ncharacter p3*3, p4*4
	parameter (p3 = nc'abcde', p4 = nc'fghi')
	ncharacter *5 p5
	parameter (p5 = nc'j' // nc'k' )
	ncharacter c10*10, c1*1

	logical x1, x2, x3, x4, x5, x6, x7, x8, x9
	parameter(x1 = nc'a' .GT. nc'b',  x2 = nc'a' .ne. nc'b')
	integer T, F, BLANK
	parameter(T = -1, F = 0, BLANK = 'A1A1'x)

	integer expect(N), rslts(N)

	data expect /	3, 102, BLANK, 6, 97, 
     +			72, BLANK, 5000, F, T,
     +			T, T, F, F, T,
     +			T, T /

	rslts(1) = len(p3) 		! 3
	rslts(2) = ichar(p4)		! 'f'
	rslts(3) = kchar(p5, 3)		! BLANK
	rslts(4) = len(p3 // p3)	! 6
	rslts(5) = kchar(p3 // p3, 4)	! 'a'

	c10 = nc'ABCD' // nc'EF' // nc'GHIJ'
	rslts(6) = kchar(c10, 8)	! 'H'
	c10 = nc'K'	! blank pad
	rslts(7) = kchar(c10, 10)	! BLANK
	c1 = nchar(5000)
	rslts(8) = kchar(c1, 1)		! 5000

	! constant fold character comparisons:

	rslts(9) = x1
	rslts(10) = x2

	x3 = nc'abcd' .LT. nc'abcxxx'		! T
	rslts(11) = x3
	x4 = LLE(nc'abc', nc'a')		! T (kanji blank pad)
	rslts(12) = x4
	rslts(13) = LGT(nc'abc', nc'abc')	! F
	rslts(14) = LGE(nc'a', nc'a\241\241\241\242')	! F
	rslts(15) = LLT(nc'a\0c', nc'a c')		! T

	rslts(16) = nc'abcd' .eq. (nc'ab' // nc'cd')	! T
	rslts(17) = nc'.\241\241\241\241\241\241\241\241' .eq. nc'.'	! T

	call check(rslts, expect, N)
	end

	integer function kchar(c, n)
	! return nth character of kanji string c.
	ncharacter*(*) c
	kchar = ichar( c(n:n) )
	end
