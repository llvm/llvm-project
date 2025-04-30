** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   ASSIGN statements and assigned GOTO's.
*   (ASSIGN of FORMAT labels is not tested).

	program p
	common j, rslts(6)
	integer j, rslts, expect(6)
	data i3 / 3 /

	assign 10 to int
	if (i3 .eq. 3)  assign 20 to int
10	goto int
	rslts(1) = 1
20	rslts(2) = 1

	assign 20 to j
	if (i3 .lt. 3)  goto j (20)
	assign 99999 to j
	if (.not. (i3 .ne. 3))  goto j, (99999, 20)
	rslts(3) = 1
99999	rslts(4) = 1

	call sub(k)

	call check(rslts, expect, 6)
	data expect / 0, 1, 0, 1, 0, 1 /

	end


	subroutine sub(k)
	common j, rslts(6)
	integer rslts

1	assign 1 to k
	assign 10 to k
	goto k (1, 10)
	rslts(5) = 1
10	rslts(6) = 1

	end
