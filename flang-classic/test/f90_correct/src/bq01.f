** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   POINTER statements, character objects

	parameter(N = 16)
	integer result(N), expect(N)
	character*8 buf
	pointer(p1, idum)
	pointer(p2, jdum)

	p1 = loc(buf)
	call sub1(p1)
	do i = 1, 8
	    result(i) = ichar(buf(i:i))
	enddo

	p2 = loc(buf)
	call sub2(p2)
	do i = 1, 8
	    result(i+8) = ichar(buf(i:i))
	enddo

	data expect/
     +    97, 98, 99,100,101,102,103,104,
     +    49, 50, 51, 52, 53, 54, 55, 56
     +  /
	call check(result, expect, N)
	end
	subroutine sub1(p1)
	character*8 ch
	pointer(p1, ch)
	ch = 'abcdefgh'
	end
	subroutine sub2(p2)
	character*4 ach(2)
	pointer(p2, ach)
	ach(2) = '5678'
	ach(1) = '1234'
	end
