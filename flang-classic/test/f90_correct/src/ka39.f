** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*--- Induction uses which are of the form <invar> - i and -i
*    

	program p
	parameter (N=8)
	integer i, j, result(N), expect(N)
	common i, j, result, expect

	data expect / 1, 2, 3, 4, -4, -3, -2, -1/

	call t1(result, 4)

	call t2(result(5), 4)

	call check(result, expect, N)
	end

	subroutine t1(iarr, iup)
	dimension iarr(iup)
	do i = 1, iup
	    iarr((iup + 1) - i) = (iup + 1) - i
	enddo
	end

	subroutine t2(iarr, iup)
	dimension iarr(-4:-1)
	do i = 1, iup
	    iarr(-i) = -i
	enddo
	end
