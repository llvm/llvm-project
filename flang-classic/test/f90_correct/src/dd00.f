** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Character assignments, including assignments to  arrays,
*   substrings, and variables, of all different kinds of
*   character expressions, including concatenations.

	program p
	character * 1 va, vb, vabc*3

	data va, vb, vabc / 'A', 'B', 'ABC' /

	call mysub(va, vabc, vb)
	end


	subroutine mysub(va, vabc, vb)
	character*(*)  va, vb, vabc
	
	character tmp*5, abcd*4, ab*2, a(2)*3
	integer i1, i2, i3, i4, i20
	external if, fx, frstu
	character*1 fx, frstu*4

	parameter(n = 26)
	character*4 rslts(n), expect(n)
        integer*4 irslts(n), iexpect(n)
        equivalence (irslts, rslts)
        equivalence (iexpect, expect)
	common /rslts/rslts
	common /expect/expect

	data abcd, ab, a / 'abcd', 'ab', '123', '456' /
	data i1, i2, i3, i4, i20 / 1, 2, 3, 4, 20 /
	data rslts / n * '????' /

	data expect / 'abcd', 'abc ', 'abcd',
     +                'ab  ', '\07 ', 'a   ',
     +                'ghij', 'rstu', 'aba ',
     +                'XXXX', 'aaba', '+*-/',
     +                'AB  ', 'ABCA', 'ABCx',
     +                'bc  ', 'Abbc', 'xXBC',
     +                'x??a', 'rAAB', '4562',
     +                '2+2+', '1*45', 'x yz',
     +                'Z CY', 'intr'          /

C --------- tests:

	rslts(1) = 'abcd'
	rslts(2) = 'ab' // 'c'
	rslts(3) = abcd

	rslts(4) = ab
	rslts(5) = char(7)
	rslts(6) = 'a'

	rslts(8) = frstu(0)
	rslts(7) = 'ghijklmnopq'
	rslts(9) = ab // char('141'o)

	rslts(10) = fx('0') // (fx('0') // fx('0')) // fx('0')
	rslts(11) = 'a' // ab // ab
	rslts(12) = expect(12)

	rslts(13) = va // vb
	rslts(14) = vabc // vabc // vabc
	rslts(15) = vabc(:) // 'xyz'

	rslts(16) = abcd(2:3)
	rslts(17) = va(i1:) // ab(2:2) // abcd(i2:)
	rslts(18) = 'x' // (fx(ab // ab) // vabc(i2:3))

	rslts(19)(:1) = 'x'
	rslts(19)(4:4) = ab
	rslts(i20)(i3:i4) = vabc // 'z'
	rslts(20)(2:2) = vabc // va
	rslts(i20)(:i1) = frstu(0)
	rslts(21)(if(1):if(4)) = a(i2)(:) // a(if(1))(2:i3)

	rslts(22)(3:4) = a(1)(if(2):if(2)) // char(ichar('+'))
	rslts(22)(1:2) = rslts(22)(3:4)
	rslts(23)(i3:i3+1) = a(i2)
	rslts(23)(i1:i1+i1) = a(1)(1:1) // ('*' // '-')
	tmp = 'x'
	tmp(i3:4) = 'yz'
	rslts(24) = tmp

	vabc(i1:2) = 'Z'
	va = 'Y'
	rslts(25) = vabc // va
	call trouble1(rslts(26))

C ----------------- check results:

	call check(irslts, iexpect, n)
	end

	subroutine trouble1(aa)
C watch for an intrinsic name declared as character and first used in
C a substring
	character*4 aa
	character*4 atan
	data i1/1/, i4/4/
	atan(i1:i4) = 'intr'
	aa = atan(:i4)
	end

C ----------------- define utility functions:

	integer function if(i)
	if = i
	return
	end

	character * 4 function frstu(i)
	frstu = 'rstu'
	end

	character *(*) function fx(i)
	character *(*) i
	fx = 'X'
	return
	end
