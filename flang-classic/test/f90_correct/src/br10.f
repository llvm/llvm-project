** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   VMS STRUCTURE/RECORD

c  test structure with and without initializations
c  test record references

	program br10
	parameter (N=6)
	integer expect(N), result(N)

	structure /stra/
	    integer a1 /1/
	    character*1 a2 /'a'/
	    integer a3 /2/
	endstructure
	
	structure /strb/
	    integer b1 /11/
	    record /stra/ attr
	    character*1 b3
	    integer b4
	endstructure

	record /strb/ recb

	result(1) = recb.attr.a1
	result(2) = ichar(recb.attr.a2)
	result(3) = recb.attr.a3
	result(4) = recb.b1

	recb.b4 = recb.attr.a1 + recb.attr.a3
	call cset(recb.b3)

	result(5) = recb.b4
	result(6) = ichar(recb.b3)

	call check(result, expect, N)

	data expect /1, 97, 2, 11, 3, 98/

	end
	subroutine cset(c)
	character*(*) c
	c = 'b'
	end
