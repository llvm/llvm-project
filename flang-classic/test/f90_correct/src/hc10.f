** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Nested IF, ELSE statements with logical expressions 
*   involving .AND., .OR., .NEQV., .EQV., and .NOT..

	parameter (N = 26)
	integer rslts(N), expect(N)
	logical t, f

	data rslts / N * 0 /
	data i1, i2, in1, i3, t, f/ 1, 2, -1, 3, .true., .false./

c                     tests 1 - 6:
	data expect / 1, 0, 0, 1, 1, 0,
c                     tests 7 - 12:
     +                0, 1, 0, 0, 0, 1,
c                     tests 13 - 18:
     +                0, 1, 0, 0, 1, 1,
c                     tests 19 - 24:
     +                0, 0, 1, 1, 0, 1,
c                     tests 25 - 26:
     +                0, 0              /
	
c ... assignments preceded by "c - t:" are the ones which should
c     be executed:

c - t:
	if (t .or. f)  rslts(1) = 1
        if (t .and.f)  rslts(2) = 1
	if (t.or.f)  goto 10
		rslts(3) = 1
10	if (t.and..not.t) goto 20
c - t:
		rslts(4) = 1


20	if (.not. i3 .le. i2) then
		if (i3 .ge. in1) then
c - t:
			rslts(5) = 1
		else if (i3 .gt. i2) then
			rslts(6) = 1
		endif
		if (t .and. t .and. t .and. (f .or. 5 .lt. i3))then
			rslts(7) = 1
		else
c - t:
			rslts(8) = 1
			if (f .eqv. t) rslts(9) = 1
		endif
	else
		rslts(9) = 2
	endif


	if ((t.neqv.t) .or. (i2.eq.i1 .eqv. t))	then
		rslts(10) = 1
	else if (f .or. f .or. f .or. .not. (i2+1.eq.i3)) then
		rslts(11) = 1
	else if ((f .or. t) .and. (t .or. t)) then
c - t:
		rslts(12) = 1
		if (.not..not.(t.neqv.f)) goto 30
			rslts(13) = 1
30		if((t .and. f) .or. (t .and. i1.eq.1)) then
c - t:
			rslts(14) = 1
			if (f .and. t) then
				rslts(15) = 1
			else if (f .or. i1 .gt. i2) then
				rslts(16) = 1
			else
c - t:
				rslts(17) = 1
c - t:
				if (t) rslts(18) = 1
				if (.not.(.not.t .eqv. in1.ge.0))then
					rslts(19) = 1
				endif
			endif

			if (f .or. i1 .eq. in1) then
				rslts(20) = 1
			elseif (.not. (t .and. f .and. t)) then
c - t:
				rslts(21) = 1
			endif
c - t:
			rslts(22) = 1
		else
			rslts(23) = 1
		endif

		if (i3 .gt. i2 .and. i2 .lt. i1)  goto 40
c - t:
			if (t .or. f .or. f)  rslts(24) = 1
			goto 50
40		rslts(25) = 1
50		continue
	else
		rslts(26) = 1
	endif


	call check(rslts, expect, N)
	end
