!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

c	Simple OpenMP Parallel Region
c	single, critical

	program p
	call t1
	end

	subroutine t1
	 parameter(n=5)
	 integer a(0:n)
	 integer result(n+4)
	 integer expect(n+4)
	 data expect/-1,0,1,2,3,-1,-1,1,4/
	 integer iam, i, omp_get_thread_num, single, critical
	 do i = 0,n
	  a(i) = -1
	 enddo
	 iam = -1
	 single = 0
	 critical = 0
c$omp	parallel private(iam)
	 iam = omp_get_thread_num()
	 if( iam .ge. 0 .and. iam .le. n ) a(iam) = iam
c$omp	single
	 single = single + 1
c$omp	end single
c$omp	critical
	 critical = critical + 1
c$omp	end critical
c$omp	end parallel
c	t1/iam should be unmodified
c	t1/a should be modified for as many threads as there are
	!print *,'iam is ',iam
	!print *,'  a is ',a
	!print *,'single is ',single,', critical is ',critical
	result(1) = iam
	do i = 0,n
	 result(i+2) = a(i)
	enddo
	result(n+3) = single
	result(n+4) = critical
	call check(result,expect,n+4)
	end
