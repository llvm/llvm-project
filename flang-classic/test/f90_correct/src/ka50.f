** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

* Recurrence recognition.

	program p
	parameter (N=23)
	common iresult(60)
	dimension iexpect(N)
	data iexpect /
     +    1, 1, 1, 1,		! /* t0:  1 - 4  */
     +    2, 2, 2, 2,		! /* t1:  5 - 8  */
     +    3, 4, 3, 4, 3, 4,	! /* t2:  9 - 14 */
     +    1, 1, 1, 4,		! /* t3: 15 - 18 */
     +    1, 2, 3, 5, 8		! /* t4: 19 - 23 */
     +  /

	iresult(1) = 1
	call t0(4)

	iresult(8) = 2
	call t1(5, 8)

	iresult(9) = 3
	iresult(10) = 4
	call t2(9, 11)

	iresult(15) = 1
	iresult(16) = 2
	call t3(15, 16)

	iresult(19) = 1
	iresult(20) = 2
	call t4(19, 21)

	call check(iresult, iexpect, N)
	end

	subroutine t0(n)
	common iresult(60)
	do i=1, n-1
	    iresult(i+1) = iresult(i)
	enddo
	end

	subroutine t1(l,u)
	integer u
	common iresult(60)
	do i=u, l+1, -1
	    iresult(i-1) = iresult(i)
	enddo
	end

	subroutine t2(l,u)
	integer u
	common iresult(60)
        do i=l, u, 2
	    iresult(i+2) = iresult(i)
	    iresult(i+3) = iresult(i+1)
	enddo
	end

	subroutine t3(l,u)
	integer u
	common iresult(60)
        do i=l, u
c	/* i+1, i is the only valid recurrence */
	    iresult(i+2) = iresult(i+1) + iresult(i)
	    iresult(i+1) = iresult(i)
	enddo
	end

	subroutine t4(l,u)
        integer u
	common iresult(60)
        do i=l, u
	    iresult(i+2) = iresult(i+1) + iresult(i)
	enddo
	end
