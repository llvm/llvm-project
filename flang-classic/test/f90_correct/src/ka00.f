** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Miscellaneous Optimizer bugs.

	program p
	parameter(n = 1)
	integer rslts(n), expect(n)
	data rslts / n * 0 /

	data i1 / 1 /

C   test 1:  top of while loop (10) target of goto:

	data expect(1) / 3 /
*if (i1 .eq. 1)  goto 10
*	return
10	if (i1 .gt. 1)  goto 20
		rslts(1) = rslts(1) + 3
		i1 = i1 + 1
		goto 10
20	continue

c   check results:

	call check(rslts, expect, n)
	end

