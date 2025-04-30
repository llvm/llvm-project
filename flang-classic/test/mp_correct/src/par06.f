!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

c	OpenMP Parallel Region
c	if clause

	program p
	call t1(1)
	call t1(0)
	end

	subroutine t1(p)
	 parameter(n=5)
	 integer a(0:n),p
	 integer result(n+2)
	 integer expect(n+2,2)
	 data expect/-1,0,1,2,3,-1,-1, -1,0,-1,-1,-1,-1,-1/
	 integer iam, i, omp_get_thread_num
	 do i = 0,n
	  a(i) = -1
	 enddo
	 iam = -1
c$omp	parallel default(none) private(iam) shared(a) if(p.gt.0)
	 iam = omp_get_thread_num()
	 if( iam .ge. 0 .and. iam .le. n ) a(iam) = iam
c$omp	end parallel
c	t1/iam should be unmodified
c	t1/a should be modified for as many threads as there are
	!print *,'iam is ',iam
	!print *,'  a is ',a
	result(1) = iam
	do i = 0,n
	 result(i+2) = a(i)
	enddo
	call check(result,expect(1,2-p),n+2)
	end
