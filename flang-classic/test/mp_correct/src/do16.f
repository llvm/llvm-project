!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!	parallel do with cyclic schedule


	program p
	 implicit none
	 integer n
	 parameter(n=10)
	 integer a(n)
	 integer result(n)
	 integer expect(n)
	 data expect/0,1,2,3,0,1,2,3,0,1/
	 integer i
	 do i = 1,n
	  a(i) = -1
	 enddo
	 call sp2(a,n)
!	 print *,a
	 do i = 1,n
	  result(i) = a(i)
	 enddo
	 call check(result,expect,n)
	end

	subroutine sp2(a,n)
	 implicit none
	 integer n
	 integer a(n)
	 integer iam, i, omp_get_thread_num
!$omp    parallel private(iam)
	  iam = omp_get_thread_num()
!$omp	  do schedule(static,1)
	   do i = 1,n
	    a(i) = iam
	   enddo
!$omp    end parallel 
	end
