** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Double precision arithmetic operations (+, -, *, /),
*   including constant folding and implied type
*   conversions of integer operands.

	programp
        implicit double precision (x)

	parameter(N = 33)
	parameter(x5dn5 = 3d-5 + 2d-5)
	double precision  rslts(N), expect(N)

c   tests 1 - 6:
	data expect /-2.0d0, 2.0d0, 3.0d0,
c   tests 7 - 22:
     +           3.5d0, 2 * x5dn5, -1.0d0, 3.5d0, 0.0d0, 4.0d0, 4.0d0,
c   tests 23 - 36:
     +           -0.5d0, -0.5d0, 5.0d0, -8.0d0, 0.5d0, 1.0d0, 2.0d0,
c   tests 37 - 50:
     +           1.5d0, -6.0d0, 1.0d0, 1.0d0, -9.0d0, 4.0d0, 4.0d0,
c   tests 51 - 66:
     +           10.0d0, -1.5d0, 4.0d0, 1.0d0,-2.5d0,4.0d0,3.0d0,1.0d0/

	data i2 / 2/, x2, xn3, x2dn5 / 2.0d0, - 3.0d0, 2.0d-5 /

c   tests 1 - 6, unary minus:

	rslts(1) = -2.0d0
	rslts(2) = - ( -x2)
	rslts(3) = -xn3

c   tests 7 - 22, addition:

	rslts(4) = 1.5d0 + 2.0d0
	rslts(5) = x2dn5 + 3d-5
	rslts(6) = 3d-5 + x2dn5
	rslts(7) = x2 + xn3

	rslts(8) = 2 + 1.5d0
	rslts(9) = xn3 + (+3)
	rslts(10) = +x2 + i2
	rslts(11) = i2 + x2

c   tests 23 - 36, subtraction:

	rslts(12) = 1.5d0 - 2.0d0
	rslts(13) = 1.5d0 - x2
	rslts(14) = x2 - xn3

	rslts(15) = (-5) - (- xn3)
	rslts(16) = i2 - 1.5d0
	rslts(17) = 3.0d0 - i2
	rslts(18) = 3.0d0 - 1

c   tests 37 - 50, multiplication:

	rslts(19) = 0.5d0 * 3.0d0
	rslts(20) = x2 * xn3
	rslts(21) = x2 * .5d0

	rslts(22) = 2 * .5d0
	rslts(23) = xn3 * 3
	rslts(24) = x2 * i2
	rslts(25) = i2 * x2

c   tests 51 - 66, division:

	rslts(26) = 5.0d0 / .5d0
	rslts(27) = xn3 / x2
	rslts(28) = x2 / .5d0
	rslts(29) = 2.0d0 / x2

	rslts(30) = 5 / (-x2)
	rslts(31) = i2 / .5d0
	rslts(32) = 3.0d0 / (i2 - 1)
	rslts(33) = 5.0d0 / 5

c   check results:

	call check(rslts, expect, 2*N)
	end
