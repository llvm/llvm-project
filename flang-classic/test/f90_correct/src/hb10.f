** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   DO loops with real and double precision index variables.

	program DO
	parameter( n = 26 )
	implicit double precision (d)
	parameter( X3$4 = (5.0 + (-0.8)) + (-0.8) )
	real rslts(N), expect(N)

	data  d2,  dn2,  x6,  x3,  x1,  xn2, i3, xnp8
     +      /2D0, -2D0, 6.0, 3.0, 1.0, -2.0,  3, -0.8  /

c       -- tests 1 - 8:
	data expect / 1.0, 2.0, 3.0, 4.1, 4.0, 6.1, 10.1, 8.1,
c       -- tests 9 - 15:
     +                0.0, 3.0, 5.0, 4.2, X3$4, 3.0, 9.0,
c       -- tests 16 - 21:
     +                0.0, 1.0, 2.0, 3.0, 20.0, 21.0,
c       -- tests 22 - 26:
     +                0.0, 4.0, 2.0, 2.0, 0.0            /

C ----------- tests 1, 2, 3, 5:    x = 1.0, 2.0, 3.0; 4.0

	i = 1
	do 10 x = 1.0, 3.0
		rslts(i) = x
		i = i + 1
10	continue
	rslts(5) = x

C ----------- tests 4, 6, 8, 7:    x = 4.1, 6.1, 8.1; 10.1

	do 20, x = 4.1, 8.3, 2.0
		i = x
20		rslts(i) = x
	rslts(7) = x

C ----------- tests 9, 10: zero trip loop  x = 3.0

	rslts(9) = 0.0
	do 30 x = x3, xn2, x1
30		rslts(9) = x
	rslts(10) = x

C ----------- tests 11, 12, 13:    x = 5.0, 4.2, 3.4; 2.6

	i = 10
	do 40 x = 5, i3, xnp8
		i = 1 + i
40		rslts(i) = x

C ----------- tests 14, 15:    y = 3.0, 5.0, 7.0; 9.0

	d = 7.9D0
	rslts(14) = 0.0
	do 50 y = i3, d, 2.0D0
50		rslts(14) = rslts(14) + 1.0
	rslts(15) = y

C ------- Double precision loops:  --------------------

C ----------- tests 16, 17, 18, 19:     d = 0.0, 1.0, 2.0; 3.0

	i = 16
	do 60, d = 0d0, .21D1
		rslts(i) = d
60		i = i + 1
	rslts(19) = d

C ----------- tests 20, 21:    d = 1.0, 3.0; 5.0

	i = 20
	rslts(22) = 0.0
	do 70, d = 1.0, i3, d2
		rslts(i) = i
70		i = 1 + i

C ----------- tests 22, 23:   (zero trip)   d = 4.0

	d = 4d0
	do 80 d = d, 5.9, dn2
80		rslts(22) = rslts(22) + 1.0
	rslts(23) = d

C ----------- tests 24, 25, 26:    d = 3.0, 5.0; 7.0

	i = 24
	rslts(26) = 0.0
	do 90, d = i3, x6 + .9, 2
		rslts(i) = 2.0
90		i = i + 1

C ----------- check  results:

	call check(rslts, expect, N)
	end
