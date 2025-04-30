!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

c	Simple OpenMP Parallel Region
c	default(private) and shared clauses
c	also default(shared) and private clauses

	program p
	call t1
	end

	subroutine t1
	 parameter(n=3)
	 integer a(0:n),b(0:n)
	 integer result(2*n+4),expect(2*n+4)
	 data expect /-1,-1,0,1,2,3,0,1,2,3/
	 integer i, iam, jam, omp_get_thread_num
	! initialize
	 do i = 0,n
	  a(i) = -1
	  b(i) = -1
	 enddo
	 iam = -1
	 jam = -1
c$omp	parallel default(private) shared(a)
	 iam = omp_get_thread_num()
	 if( iam .ge. 0 .and. iam .le. n ) a(iam) = iam
c$omp	end parallel
c$omp	parallel default(shared) private(jam)
	 jam = omp_get_thread_num()
	 if( jam .ge. 0 .and. jam .le. n ) b(jam) = jam
c$omp	end parallel
	!print *,'iam is ',iam
	!print *,'  a is ',a
	!print *,'jam is ',jam
	!print *,'  b is ',b
	result(1) = iam
	result(2) = jam
	do i = 0,n
	 result(3+i) = a(i)
	 result(3+i+n+1) = b(i)
	enddo
	call check(result,expect,2*n+4)
	end
