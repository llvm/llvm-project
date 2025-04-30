!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

c	Simple OpenMP Parallel Region
c	private atomic

	program p
	call t1
	end

	subroutine t1
	 parameter(n=5)
	 integer a(0:n)
	 integer result(n+3)
	 integer expect(n+3)
	 data expect/-1,0,1,2,3,-1,-1,6/
	 integer iam, i, omp_get_thread_num, atomic, f
	 external f
	 do i = 0,n
	  a(i) = -1
	 enddo
	 iam = -1
	 atomic = 0
c$omp	parallel private(iam)
	 iam = omp_get_thread_num()
	 if( iam .ge. 0 .and. iam .le. n ) a(iam) = iam
c$omp	atomic
	 atomic = atomic + f(iam)
c$omp	end parallel
c	t1/iam should be unmodified
c	t1/a should be modified for as many threads as there are
	!print *,'iam is ',iam
	!print *,'  a is ',a
	!print *,'atomic is ',atomic
	result(1) = iam
	do i = 0,n
	 result(i+2) = a(i)
	enddo
	result(n+3) = atomic
	call check(result,expect,n+3)
	end

	integer function f(i)
	f = i
	end
