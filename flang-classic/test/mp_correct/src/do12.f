!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!	OpenMP Parallel Region in caller, dynamic schedule in callee

! If -Mfprelaxed is used, the test could fail because of the use of
! the rsqrt instruction when computing sqrt.
! The following -y says to inhibit '-Mfprelaxed'; need this because
! That doesn't work for IPA case, though.  Change the op to use double.
!!!pgi$g -y 15 0x10
	program p
	 implicit none
	 integer n
	 parameter(n=10)
	 real a(0:n),b(n),y(n)
	 real result(2*n)
	 real expect(2*n)
	 data expect/1.,4.,9.,16.,25.,36.,49.,64.,81.,100.,
     &		     1.,2.,3.,4.,5.,6.,7.,8.,9.,10./
	 integer i
	 do i = 0,n
	  a(i) = i*i+i
	 enddo
!$omp   parallel
	 call sp2(a,b,y,n)
!$omp   end parallel
	 !print *,b
	 !print *,y
	 do i = 1,n
	  result(i) = b(i)
	  result(n+i) = y(i)
	 enddo
	 call check(result,expect,2*n)
	end

	subroutine sp2(a,b,y,n)
	 implicit none
	 integer n
	 real a(0:n),b(n),y(n)
	 integer i
!$omp   do schedule(dynamic,10)
	do i = 1,n
	 b(i) = (a(i) + a(i-1)) / 2.0
	enddo
!$omp   end do
!$omp   do schedule(dynamic)
	do i = 1,n
	 y(i) = dsqrt(dble(b(i)))
	enddo
!$omp   end do nowait
	end
