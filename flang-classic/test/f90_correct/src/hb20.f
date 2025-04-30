** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   DO loops - special cases:
*      (1) control flow statements as last statement.

	parameter(n = 6)
	integer rslts(n), expect(n)

	do 10 j = 1, 1
		do 10 i = 1, 4
	data (expect(i), i = 1, n) / 2, -2, 4, -4, 101, 104 /
			if (and(i, 1)) then
				rslts(i) = i + 1
			else
				rslts(i) = - i
10			endif


	rslts(5) = 0
	do 20 i = 1, '7fffffff'x
		rslts(5) = rslts(5) + 1
20		goto(30, 40) i + 1
30	rslts(5) = rslts(5) + 10
40	rslts(5) = rslts(5) + 100


	rslts(6) = 0
	do 50 i = -999999, -999991
		rslts(6) = rslts(6) + 1
50		call sub(*60, *70)
60	rslts(6) = rslts(6) + 10
70	rslts(6) = rslts(6) + 100


	call check(rslts, expect, n)
	end

c-------------

	subroutine sub(*, *)
	save c
	integer c
	data c / 0 /
	c = c + 1
	if (c .eq. 4)  return 2
	end
