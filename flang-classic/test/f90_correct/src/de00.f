** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Character expressions as subprogram arguments.

	program p
	call mysub('a', 'XYZ', '3456789')
	return
	end

	subroutine mysub(ca, cxyz, c39)
	character*(*) ca, c39, cxyz*3

	parameter(n=37)
	integer irslts(n), expect(n)
	character*4 crslts(n)
	equivalence(irslts, crslts)

	character*3 cf, csf, c2*2, loc1*1, loc6*6, loca(4)*2

	csf(c2) = '.' // c2
	data loc1, loc6, loca / 'm', 'nopqrs', 'bc', 'de', 'fg', 'hi'/
	data i2, i3 / 2, 3 /

c --- test 1:

	data expect(1) / 11 /
c         ca and c39 have passed-length 4:
	irslts(1) = len(ca // c39 // cxyz )

c --- tests 2 - 7:

	data (expect(i), i = 2, 7) / 'a   ', '3456', 'XYZ ', 1, 7, 3 /
	call sub1(crslts(2), ca, c39, cxyz, irslts(5))

c --- tests 8 - 13:

	data (expect(i), i = 8, 13)/ 'aXYZ', '4567', '5678', 4, 4, 4 /
	call sub1(crslts(8), 'a'//cxyz, c39(2:5), c39(i3:2*i3),
     +                                                   irslts(11))

c --- tests 14 - 19:

	data (expect(i), i = 14,19)/ '\05 ', 'Acde', 'YZ  ', 1, 4, 3 /
	call sub1(crslts(14), char(5), 'A' // ('c'//'de'), cf(cxyz),
     +                                                    irslts(17))

c --- tests 20 - 25:

	data (expect(i), i = 20,25)/ 'aYZ ', '-bcd', 'fg  ', 3, 5, 2 /
	call sub1(crslts(20), ca(1:1) // cxyz(i2:i3),
     +              '-' // cf('abcde') // '+', loca(i3), irslts(23))

c --- tests 26 - 31:

	data (expect(i), i = 26,31)/ 'mXYZ', '4eee', 'Z   ', 6, 4, 3 /
	call sub1(crslts(26), loc1 // cxyz // loca(4),
     +              c39(i2:i2) // 'eee', cf(cf(cxyz)), irslts(29))

c --- tests 32 - 37:

	data (expect(i), i = 32,37)/ 'bc  ', '.89 ', '2342', 2, 3, 6 /
	call sub1(crslts(32), loca, csf(c39(6:7)),
     +              cf('12345') // cf('12345'), irslts(35))

c --- check results:

	call check(irslts, expect, n)
	end


	subroutine sub1(crslts, c1, c2, c3, irslts)
	character*(*) c1, c2, c3, crslts(3)
	integer irslts(3)

	crslts(1) = c1
	crslts(2) = c2
	crslts(3) = c3

	irslts(1) = len(c1)
	irslts(2) = len(c2)
	irslts(3) = len(c3)

	end


	function cf(c)
	character*(*) cf, c
	cf = c(2:)
	end
