** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   KANJI - DATA statements using NCHARACTER.

	program p
	parameter(N=23)

	ncharacter*1 a1, b1, c1(4)
	ncharacter*2 a2
	ncharacter*3 a3
	ncharacter*5 a5(6), b5, c5(4)

	common /cblock/ a2, c5

	integer expect(N), rslts(N), BLANK
	parameter(BLANK = 'A1A1'x)		! kanji blank

	data i1, i6, i4 / 1, 6, 4/
	data expect /	97, 99, 101, 98, BLANK,	    ! 'a', 'c', 'e', 'b', ' ',
     +			102, 103, 104, 105, BLANK,  ! 'f', 'g', 'h', 'i', ' ',
     +			BLANK, 105, BLANK, 106, 107,! ' ', 'i', ' ', 'j', 'k',
     +			107, BLANK, 108, 109, 110,  ! 'k', ' ', 'l', 'm', 'n',
     +			114, 115, 119 /		    ! 'r', 's', 'w' /

	data a1/nc'a'/, a2/nc'b'/
	data b1/nc'cd'/, c1(2)/nc'e'/
	data a3(1:1)/nc'f'/, a3(2:3)/nc'gh'/
	data a5(6), (a5(i), i = 1, 5) / 2*nc'i', 2*nc'j', 2*nc'k'/
	data (b5(i:i+1), i = 1, 4, 2) / nc'lm', nc'np'/
	data c5(2)(2:3), c5(4)(5:5) / nc'qr', nc's'/
	data c5(4)(1:4) / nc'tuvw' /

	rslts(1) = ichar(a1)		!  'a'
	rslts(2) = ichar(b1)		!  'c'
	rslts(3) = ichar(c1(2))		!  'e'
	rslts(4) = ichar(a2(1:1))	!  'b'
	rslts(5) = ichar(a2(2:2))	!  ' '

	rslts(6) = ichar(a3(i1:i1))	!  'f'
	rslts(7) = ichar(a3(2:2))	!  'g'
	rslts(8) = ichar(a3(3:3))	!  'h'
	rslts(9) = ichar(a5(i6)(1:1))	!  'i'
	rslts(10) = ichar(a5(6)(i1*2:2))!  ' '

	rslts(11) = ichar(a5(6)(5:5))	!  ' '
	rslts(12) = ichar(a5(1)(1:1))	!  'i'
	rslts(13) = ichar(a5(2)(2:2))	!  ' '
	rslts(14) = ichar(a5(3)(1:1))	!  'j'
	rslts(15) = ichar(a5(i4)(1:1))	!  'k'

	rslts(16) = ichar(a5(5)(1:1))	!  'k'
	rslts(17) = ichar(a5(5)(5:5))	!  ' '
	rslts(18) = ichar(b5(1:1))	!  'l'
	rslts(19) = ichar(b5(2:2))	!  'm'
	rslts(20) = ichar(b5(3:3))	!  'n'

	rslts(21) = ichar(c5(2)(3:3))	!  'r'
	rslts(22) = ichar(c5(4)(5:5))	!  's'
	rslts(23) = ichar(c5(4)(4:4))	!  'w'

	call check(rslts, expect, N)
	end
