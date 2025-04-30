** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Optimizer tests of converting while loops to "do while"

	program ka01
	parameter(N=7)
	common iresult(N), iexpect(N)
	data iresult /N*0/
	call test1_2
        call test3_7
        call check(iresult, iexpect, N)
	data iexpect /
     +    4, 5,                         ! tests 1-2
     +    4, 4, 4, 4, 4                 ! tests 3-7
     +  /
	end

	subroutine incr(i)
	parameter(N=7)
	common iresult(N), iexpect(N)
	iresult(i) = iresult(i) + 1
	end

	subroutine test1_2()
	i = 1
	do 100 while (i .le. 4)
	    call incr(1)
	    i = i + 1
100	continue
	do while (i .gt. 0)
	    call incr(2)
	    i = i - 1
	enddo
	end

	subroutine test3_7()
	i = 4
	do 100 while (i .ge. 1)
	    k = 4
	    call incr(3)
	    do while (k .gt. 0)
		j = k
		call incr(3 + j)
		k = k - 1
	    enddo
	    i = i - 1
100     continue
	end
