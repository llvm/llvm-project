!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!	block-cyclic schedule with variable block size

	program p
	 implicit none
	 integer n
	 parameter(n=10)
	 real a(0:n),b(n),x(n),y(n)
	 real result(2*n)
	 real expect(2*n)
	 data expect/1.,3.,5.,7.,9.,11.,13.,15.,17.,19.,
     &		     1.,2.,3.,4.,5.,6.,7.,8.,9.,10./
	 integer i
	 do i = 0,n
	  a(i) = 2*i
	 enddo
	 do i = 1,n
	  x(i) = i*i
	 enddo
	 call sp2(a,b,x,y,n,2)
	 !print *,b
	 !print *,y
	 do i = 1,n
	  result(i) = b(i)
	  result(n+i) = y(i)
	 enddo
	 call check(result,expect,n)
	end

	subroutine sp2(a,b,x,y,n,m)
	 implicit none
	 integer n,m
	 real a(0:n),b(n),x(n),y(n)
	 integer i
!$omp   parallel
!$omp   do schedule(static,m)
	do i = 1,n
	 b(i) = (a(i) + a(i-1)) / 2.0
	enddo
!$omp   end do
!$omp   end parallel
	end
