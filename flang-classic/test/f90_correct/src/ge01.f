** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Double complex arithmetic operations (+, -, *, /), including constant
*   folding and implied type conversions to double complex.

	implicit double complex (c, p)
	parameter(n = 35)
	double complex rslts(n), expect(n)
	integer*8 iexpect(12), iex2(4)
	equivalence (iexpect, expect(19)), (iex2, expect(33))

	common c35, c2462, c79

	parameter(p1 = - (1, 2))
	parameter(p2 = (1, 2) + (2.0, 3))
	parameter(p3 = (1.0, 1.0) - (-5.0))
	parameter(p4 = (-2, 3) * (4, 5))
	parameter(p5 = (4, 6) / 2 )

	data i12, i4 / 12, 4 /
	data x6      / 6.0   /
	data c12, cn46 / (1, 2), (-4.0, 6) /

c --- tests 1 - 12:  addition:

	data (expect(i), i = 1, 6) / (-4, 6), (11, -2), (9, -8),
     +                               (-3, 8), (13, 2), (14, 8)   /

	rslts(1) =  (1.0, -2.0) + (-5.0, 8.0)
	rslts(2) = (1.0, -2.0) + 10
	rslts(3) = dcmplx(8, -9) + (1, 1)
	rslts(4) = c12 + cn46
	rslts(5) = i12 + c12
	rslts(6) = c12*4 + 10.0

c --- tests 13 - 24:  subtraction:

	data (expect(i), i = 7, 12) / (-6, 10), (9, -2), (-6, -8),
     +                                (-20, 16), (-5, 2), (2, -8) /

	rslts(7) = (-5.0, 8.0) - (1.0, -2.0)
	rslts(8) = 10.0d0 - (1.0, 2.0)
	rslts(9) = (2, -2) - dcmplx(8.0, x6)
	rslts(10) = (cn46 - c12) * i4
	rslts(11) = c12 - x6
	rslts(12) = x6 - (i4 * c12)

c --- tests 25 - 36:  multiplication:

	data (expect(i), i = 13, 18) / (26, 51), (10, 15), (-14, 22),
     +                                 (-16, -2), (-4, 6), (30, 12)  /

	rslts(13) = (2.0, -5.0) * (-7.0, 8.0)
	rslts(14) = (-2, -3) * (-5)
	rslts(15) = dcmplx(x6, 2.0) * dcmplx(-1, i4)
	rslts(16) = c12 * cn46
	rslts(17) = (1, 0) * cn46
	rslts(18) = (i4 + c12) * x6

c --- tests 37 - 48:  division:

	data iexpect / 1, -3,   3, -6,    3, -3,
     +                 1, -3,   3, 4,     1, -3  /

	data c973/(.9, 7.3)/,  c6181/(6.1, 8.1)/,  c2n1/(2,-1)/

	call ctoi(rslts(19), (.9d0, 7.3d0) / (-2d0, 1.0d0))
	call ctoi(rslts(20), (6.1d0, -12.1d0) / 2d0)
	call ctoi(rslts(21), (6.1d0) / (1d0, 1d0))
	call ctoi(rslts(22), c973 / (-2d0, 1.0d0))
	call ctoi(rslts(23), c6181 / (x6 - 4.0d0) )
	call ctoi(rslts(24), c973 / (-c2n1) )

c --- tests 49 - 54:  unary minus:

	data (expect(i), i = 25, 27) / (-1, -2), (-2, 1), (14, 4) /

	rslts(25) = - (1.0, 2.0)
	rslts(26) = - c2n1
	rslts(27) = 2.0 * (-(-(c12 + x6)))

c --- tests 55 - 64:  double complex parameters:

	data (expect(i), i=28, 32) / (-1, -2), (3, 5), (6, 1),
     +                               (-23, 2), (2, 3)         /

	rslts(28) = p1
	rslts(29) = p2
	rslts(30) = p3
	rslts(31) = p4
	rslts(32) = dcmplx(nint(real(p5)), nint(aimag(p5)) )

c --- tests 65 - 70:  double complex multiplication/division bugs:

	data iex2 / 7, 9, 3, 5 /, expect(35) / (-11, 23) /
	data c35, c79, c2462 / (3, 5), (7, 9), (-24, 62) /

	c35 = c2462 / c35
	call ctoi(rslts(33), c35)
	c2462 = c2462 / c79
	call ctoi(rslts(34), c2462)
        c79 = c79 * c12
        rslts(35) = c79

c --- check results:

	call check(rslts, expect,  4*n)
	end

cccccccccccccc

	subroutine ctoi(ia, c)
	double complex c
	integer*8 ia(2)
	ia(1) = real(c) + .001
	ia(2) = aimag(c) + .001
	end
