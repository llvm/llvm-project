** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Function subprograms - all function return types are tested
*   and use of function name in various contexts within function
*   body.  ENTRY statements are not tested.

c things tested include:
c  (1) functions with all possible return types.
c  (2) functions with 0, 1, and more arguments.
c  (3) declaration of function type via FUNCTION statement,
c      type spec. statement, and IMPLICIT statement.
c  (4) Use of function name on left and right of assignment
c      statements, as actual argument, with substring specifier.

	program p
	parameter(n = 16)
	integer TRUE
	parameter(TRUE = -1)
	integer rslts(n), expect(n)
	logical lrslts(6:16)
	equivalence (lrslts, rslts(6))

	integer f1
	integer*2 f2
	real f3
	real*8 f4
	logical f5
	external f1, f3, f6, f11, cadd2
	complex f6
	double precision f7
	character*1 f8, f12
	character*4 f9
	character*6 f10
	character*3 f11
	character c1*1, c4*4

c  ----------- tests 1 - 9:  non-character functions:

	rslts(1) = f1() + 4
	rslts(2) =  f2(-5)
	rslts(3) = f3() + 0.2
	rslts(4) = 2 * f4(3.6d0)
	rslts(5) = 0
	if (f5(2, -2))  rslts(5) = 2
	lrslts(6) = f5(0, 0) .neqv. .false.
	rslts(7) = f6()
	rslts(8) = aimag( f6() )
	rslts(9) = -f7(3.1d0)

c  ----------- tests 10 - 16:   character functions:
	
	rslts(10) = ichar( f8() )
	lrslts(11) = 'abcd' .eq. f9()
	c4 = f9()
	rslts(12) = ichar( c4(3:3) )
	lrslts(13) = f10() .eq. 'ABCCFE'
	lrslts(14) = 'ABCDEF' .eq. f10()
	lrslts(15) = f11('mno', 3) .eq. 'mnq'
	data c1 /'g' /
	rslts(16) = ichar( f12(c1) )

c  ------------ check results:

	call check(rslts, expect, n)
	data expect / 12, 5, 6, 7, 2, 0, -1, -1, 1,
     +                7, TRUE, 99, TRUE, 0, TRUE, 107        /
	end

	integer function f1()
	f1 = 8
	end

	integer*2 function f2(i)
	f2 = -i
	return
	end

	function f3()
	f3 = 5.9
	end

	function f4(d)
	implicit double precision (d, f)
	f4 = d
	return
	end

	logical function f5(a, b)
	implicit complex (f), integer (a, b)
	f5 = a .eq. b
	f5 = .not. f5
	end

	function f6()
	complex f6, plus1
	f6 = (1.0, 2.0)
	f6 = - f6
	f6 = plus1( f6 )
	return
	end

	complex function plus1(c)
	complex c
	plus1 = c + (0.0, 1.0)
	end

	function f7(d)
	real * 8 f7, d, f4
	f7 = - f4(d) + 1.0
	call dincr(f7)
	end

	subroutine dincr(d)
	double precision d
	d = d + 1.0
	end

	character*1 function f8()
	f8 = '\07'
	return
	end

	function f9()
	character*(*) f9
	f9 = 'abcd'
	end

	function f10()
	implicit character*(*) (f)
	character*3 c
	f10 = 'ABCDEF'
	c = f10(3:5)
	f10(4:6) = c
	call cadd2(f10(5:5))
	end

	subroutine cadd2(c)
	character*1 c
	c = char( ichar(c) + 2 )
	return
	end

	character*(*) function f11(a, b)
	character*3 a, c
	integer b
	f11 = a
	c = f11
	f11(3:) = char( ichar(c(3:3))+2 )
	return
	end

	function f12(a)
	implicit character*1 (a-z)
	call cadd2(a)
	f12 = a
	call cadd2(f12)
	end
