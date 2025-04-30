** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   IF, ELSEIF, ELSE, and ENDIF statements with simple 
*   logical conditions and no nesting.

	parameter(N = 13)
        implicit logical (t, f)
	integer rslts(N), expect(N)

	data rslts / N * 0 /,
     +       expect/ 2, 0, 2, 10, 1, 10, 100, 1, 0, 0, 2, 8, 0/

	data i1, i2, i3, i5, t, f / 1, 2, 3, 5, .true., .false. /

	if (i2 .gt. i1)  then
		rslts(1) = 2
	endif
	if (i3 .lt. i2)  then
		rslts(2) = 4
	endif

	if (i3 .eq. i2 + 1)  then
		rslts(3) = 2
	else
		rslts(3) = rslts(3) + 1
	endif
	if (i2 + 1 .ne. i3)  then
		rslts(4) = rslts(4) + 1
	else
		rslts(4) = rslts(4) + 10
	endif

	if (i3 .ge. i3) then
		rslts(5) = 1
	elseif (i3 .ge. i3) then
		rslts(5) = rslts(5) + 10
	else
		rslts(5) = rslts(5) + 100
	endif
	if (i3 .le. i2)then
		rslts(6) = 1
	elseif(i2 .le.i3)then
		rslts(6) = rslts(6) + 10
	else
		rslts(6) = rslts(6) + 100
	endif
	if (f)then
		rslts(7) = 1
	else if (.not. t) then
		rslts(7) = rslts(7) + 10
	else
		rslts(7) = rslts(7) + 100
	endif

	if (i3-1 .eq. 2)   rslts(8) = 1
	if (i3.ge.i2+i2)   rslts(9) = 1
	if(i2 .ne. i2+1)goto10
		rslts(10) = 2
10	if (i2 .gt. i3) goto  20
		rslts(11) = 2

20	if (-i2 .ge. i2) then
		rslts(12) = 1
	else if (i2 .ne. i1 * i2) then
		rslts(12) = rslts(12) + 2
	else if (i2 .lt. -(-i2)) then
		rslts(12) = rslts(12) + 4
	else if (i2 - i2 .eq. i3 - i1 - i2) then
		rslts(12) = rslts(12) + 8
	endif

	if (i3 .eq. i2) then
		rslts(13) = 1
	elseif (i3 .eq. i1) then
		rslts(13) = rslts(13) + 2
	endif

c   check results:

	call check(rslts, expect, N)
	end
