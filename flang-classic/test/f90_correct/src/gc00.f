** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Real arithmetic operations (+, -, *, /) including constant
*   folding and implied type conversions of integer operands.

	programp

	parameter(N = 33)
	real rslts(N), expect(N)

c   tests 1 - 3:
	data expect /-2.0, 2.0, 3.0,
c   tests 4 - 11:
     +               3.5, 5e-5, 5e-5, -1.0, 3.5, 0.0, 4.0, 4.0,
c   tests 12 - 18:
     +               -0.5, -0.5, 5.0, -8.0, 0.5, 1.0, 2.0,
c   tests 19 - 25:
     +               1.5, -6.0, 1.0, 1.0, -9.0, 4.0, 4.0,
c   tests 26 - 33:
     +               10.0, -1.5, 4.0, 1.0, -2.5, 4.0, 3.0, 1.0 /

	data i2 / 2/, x2, xn3, x2en5 / 2.0, - 3.0, 2.0e-5 /

c   tests 1 - 3, unary minus:

	rslts(1) = -2.0
	rslts(2) = - ( -x2)
	rslts(3) = -xn3

c   tests 4 - 11, addition:

	rslts(4) = 1.5 + 2.0
	rslts(5) = x2en5 + 3e-5
	rslts(6) = 3e-5 + x2en5
	rslts(7) = x2 + xn3

	rslts(8) = 2 + 1.5
	rslts(9) = xn3 + (+3)
	rslts(10) = +x2 + i2
	rslts(11) = i2 + x2

c   tests 12 - 18, subtraction:

	rslts(12) = 1.5 - 2.0
	rslts(13) = 1.5 - x2
	rslts(14) = x2 - xn3

	rslts(15) = (-5) - (- xn3)
	rslts(16) = i2 - 1.5
	rslts(17) = 3.0 - i2
	rslts(18) = 3.0 - 1

c   tests 19 - 25, multiplication:

	rslts(19) = 0.5 * 3.0
	rslts(20) = x2 * xn3
	rslts(21) = x2 * .5

	rslts(22) = 2 * .5
	rslts(23) = xn3 * 3
	rslts(24) = x2 * i2
	rslts(25) = i2 * x2

c   tests 26 - 33, division:

	rslts(26) = 5.0 / .5
	rslts(27) = xn3 / x2
	rslts(28) = x2 / .5
	rslts(29) = 2.0 / x2

	rslts(30) = 5 / (-x2)
	rslts(31) = i2 / .5
	rslts(32) = 3.0 / (i2 - 1)
	rslts(33) = 5.0 / 5

c   check results:

	call check(rslts, expect, N)
	end
