** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

* Constant prop. - call in path problems

	program p
	parameter (N=1)
	common iresult(N)
	dimension iexpect(N)
	data iexpect /
     +    9			! t0
     +  /

	iresult(1) = 0
	call t0(iresult(1), 5)

	call check(iresult, iexpect, N)
	end

	subroutine t0(ii, n)
	jj = 1			!jj cannot be copied
	do i = 1, n
	    ii = jj + ii
	    call set(jj)	!jj potentially modified
	enddo
	end
	subroutine set(jj)
	jj = 2
	end
