!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!	OpenMP do ordered


	program p
	 implicit none
	 integer n
	 parameter(n=10)
	 integer a(n)
	 integer result(n)
	 integer expect(n)
	 data expect/1,2,3,4,5,6,7,8,9,10/
	 integer i
	 do i = 1,n
	  a(i) = -1
	 enddo
	 call sp2(a,n)
	 !print *,a
	 do i = 1,n
	  result(i) = a(i)
	 enddo
	 call check(result,expect,n)
	end

	subroutine sp2(a,n)
	 implicit none
	 integer n
	 integer a(n)
	 integer i
!$omp    parallel
!$omp	  do ordered
	   do i = 1,n
	     call work(a, i)
	   enddo
!$omp    end parallel 
	end
	subroutine work(a,ii)
	integer a(*)
	integer idx
	save idx
	data idx/1/
!$omp ordered
!$omp critical
	a(idx) = ii
	idx = idx + 1
!$omp endcritical
!$omp endordered
	end
