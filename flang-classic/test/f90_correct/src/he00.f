** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Arithmetic IF statements.

	program p
	parameter(n = 31)
	integer rslts(n), expect(n)
	double precision dx

	data rslts / n * 0 /

c ---------- tests 1 - 3: integer arithmetic IF:

	do 10 i = -1, 1, 1
		if (i) 11, 12, 13
		rslts(i+2) = 1
11		rslts(i+2) = rslts(i+2) + 2
		goto 10
12		rslts(i+2) = rslts(i+2) + 4
		goto 10
13		rslts(i+2) = rslts(i+2) + 8
10	continue

c ----------- tests 4 - 6: real arithmetic IF:

	do 20 x = 0, 480.0, 239.3
		if (x - 239.3)  23, 22, 21
21		rslts(4) = x
		goto 20
22		rslts(5) = x
		goto 20
23		rslts(6) = x
20	continue

c ------------ tests 7 - 11:  double precision arithmetic IF:

	data dx / 236.78D17 /

	do 30 i = 1, 5
		if (dble(i-3) * dx) 31, 32, 33
31		rslts(i+6) = 1
		goto 30
32		rslts(i+6) = 2
		goto 30
33		rslts(i+6) = 3
30	continue

c ------------- tests 12 - 18: arithmetic IFs with duplicate labels:

	data i1, i2 / 1, 2 /

	if (i1) 40, 40, 40
40	if (i1) 50, 50, 60
50	rslts(12) = 1
60	rslts(13) = 1
	if (i1 - i2) 51, 51, 61
51	rslts(14) = 1
61	rslts(15) = rslts(15) + 1
	if (i1 + i1 - i2) 61, 62, 61
62	rslts(16) = 1
	if (i2*8) 63, 63, 64
63	rslts(17) = 1
64	rslts(18) = 1

c -------------- tests 19 - 28: Constant arithmetic values:

	if (-6) 70, 71, 72
70	rslts(19) = 1
71	if (99999 ) 71, 73, 74
72	rslts(20) = 1
73	rslts(21) = 1
74	rslts(22) = 1
	if ('0'o) 76, 75, 76
75	rslts(23) = 1
	goto 77
76	rslts(24) = 1
77	if (-i2) 78, 79, 79
79	rslts(25) = 1
78	rslts(26) = 1
	if (i2*2 - i1*4) 80, 81, 81
80	rslts(27) = 1
81	rslts(28) = 1

c ----------- tests 29 - 31: real arithmetic IF with func calls:

	do 200 x = 0, 480.0, 239.3
		if (foo(x) - 240.3)  230, 220, 210
210		rslts(29) = x
		goto 200
220		rslts(30) = x
		goto 200
230		rslts(31) = x
200	continue

c --------------- check results:

	call check(rslts, expect, n)
	data expect / 2, 4, 8,
     +                478, 239, 0,
     +                1, 1, 2, 3, 3,
     +                0, 1, 1, 1, 1, 0, 1,
     +                1, 0, 0, 1, 1,  0, 0, 1, 0, 1,
     +		      478, 239, 0 /
	end

	function foo(z)
	foo = z + 1.0
	return
	end

