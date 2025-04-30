** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Alternate returns.

	program p
	parameter(n = 13)
	integer rslts(n), expect(n)

	external s1, s5, s8

	data rslts / n * 0 /

	do 10 i = 1, 3
		call s1(*20)
		rslts(1) = rslts(1) + i
		goto 10
20		rslts(2) = rslts(2) + i
10	continue

	do 30 i = 1, 6
		call s2(*31, *99999, *31, *99999)
		rslts(3) = rslts(3) + i
		goto 30
31		rslts(4) = rslts(4) + i
		goto 30
99999		rslts(5) = rslts(5) + i
30	continue

	call s4(0, *40)
	rslts(6) = 1
40	rslts(7) = 1
	call s4(1, *41)
	rslts(8) = 1
41	rslts(9) = 1

	x = -1.0
50	x = x + 1.0
	call s5(*50, x)
	rslts(10) = int(x + .1)
	call s6(-2.3, *50, *51)
	rslts(11) = 1
51	rslts(12) = rslts(12) + 1
	call s6(x+20, *50, *51)

	call s7
	call s8(*60)
	rslts(13) = 1

60	call check(rslts, expect, n)

	data expect / 4, 2, 7, 8, 6,
     +                1, 1, 0, 1, 1, 0, 1, 1 /
	end

cccccccccccccccccccccccccccccccccccccccccccccccccccc

	integer function if(i)
	if = i
	end

	subroutine s1(*)
	common /x/ i
	data i /-1/
	i = i + 1
	return i
	end

	subroutine s2(*,*,*,*)
	common /coms2/ i
	data i /0/
	i = i + 1
	return 6 - i
	end

	subroutine s4(i, *)
	if (i .eq. 1)   return 1
	j = if(1)
	return
	end

	subroutine s5(*, x)
	entry s6(x, *, *)
	if (x .eq. 0)  return if(1)
	if (x .gt. 0)  return
	return 2
	end

	subroutine s7
	entry s8(*)
	if (if(1) .eq. 1)  return
	return 1
	end
