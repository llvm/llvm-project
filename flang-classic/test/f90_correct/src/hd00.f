** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Computed GOTO statements.

	program p
	parameter(n = 14+2)
	integer rslts(n), expect(n), j(1)

	data rslts / n * 0 /
	data expect / 9, 3, 5, 4, 
     +                10, 10, 11,
     +                800000010, 76003000, 500000, 40200,
     +                9, 5, 3,
     +                3, 5     /

c  ----------------- tests 1 - 4:

	do 10 i = 1, 6
		goto (11, 12, 13)  i - 2
			rslts(1) = rslts(1) + i
			goto 10
11			rslts(2) = rslts(2) + i
			goto 10
13			rslts(3) = rslts(3) + i
			goto 10
12			rslts(4) = rslts(4) + i
10	continue

c  ------------------ tests 5 - 7:

	data  j / 1/
	goto (21) , j(1)
	rslts(5) = rslts(5) + 1
21	rslts(5) = rslts(5) + 10

	goto (22, 23), 2 * j(1)
22	rslts(6) = rslts(6) + 1
23	rslts(6) = rslts(6) + 10

	goto (25, 26) 1
	goto 26
25	rslts(7) = rslts(7) + 1
26	rslts(7) = rslts(7) + 10

c  ------------------- tests 8 - 11:

	k = 1
	kk = 0
	do 30 i = 8, 1, -1
		k = k * 10
		kk = kk + 1
		goto (31, 31, 32, 33, 31, 33) i - 1
		rslts(8) = rslts(8) + k * kk
		goto 30
31		rslts(9) = rslts(9) + k * kk
		goto 30
32		rslts(10) = rslts(10) + k * kk
		goto 30
33		rslts(11) = rslts(11) + kk * k
30	continue

c  -------------------- tests 12 - 14:

	data i1 /1/
	if (i1 .eq. 1)  goto (40, 41) , i1
	rslts(12) = 100
40	rslts(12) = rslts(12) + 1
41	rslts(12) = rslts(12) + 8

42	i1 = i1 + 1
	rslts(13) = rslts(13) + i1
	goto (43, 42) ifunc(i1)
	rslts(14) = rslts(14) + i1
	goto 44
43	stop "test 14 error"

c  -------------------- part 2: computed gotos in inlinable functions:
c  -------------------- tests 15, 16:

44	continue
	call t15( rslts(15) )
	call t16( rslts(16), 4)

c  -------------------- check results:

	call check(rslts, expect, n)
	end

	integer function ifunc(i)
	ifunc = i
	end

	subroutine t15(r)
	integer r

	goto (10, 20)	ifunc(2)
10	r = -2
	return
20	r = 3
	end

	subroutine t16(r, k)
	integer r

60	continue
50	continue
	r = -8
	goto (20, 50, 30, 40, 10, 60, 110, 70, 80, 90, 100, 120)  k
	r = -7
	return
10	r = -3
20	r = -3
30	r = -3
	return
40	r = 5		! should come here.
	return
120	continue
110	continue
100	continue
	r = -4
90	continue
80	continue
70	continue
	end
