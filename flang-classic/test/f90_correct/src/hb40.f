** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   DO WHILE statement (VMS).

	program hb40
	integer n, rslts, expect
	parameter (n = 4)
	dimension rslts(n), expect(n)
	real x(3), y(3)

	data expect / 0, 40, 10, 43 /

	rslts(1) = 0
	dowhile (.false.)
		rslts(1) = 10
	enddo

	rslts(2) = 10
	do 10 while (.true.)
		rslts(2) = rslts(2) + rslts(2)
10	if (rslts(2) .gt. 30)  goto 11
11	continue

	data i10 / 10 /
	do 30, WHILE(i10.ne.10)
		i10 = i10 + 1
30	continue
	rslts(3) = i10

	do40while (0 .eq. 1)
40	enddo

	data x / 1.0, 2.0, 3.0 /
	data y / 3.0, 5.0, 7.0 /
	sum = 0
	do i = 1, 3
		do
     +		while (sum .lt. 10 * i)
			sum = sum + x(i) * y(i)
		enddo
	enddo
	rslts(4) = sum

	call check(rslts, expect, n)
	end
