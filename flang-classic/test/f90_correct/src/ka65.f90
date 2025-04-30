!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! allocate temp for the assignment in 's' of the proper size

	subroutine s(a,n,m)
	 real a(:)
	 a(n-1:n-m:-2) = a(1:m:2)
	end

	program p
	 interface
	  subroutine s(a,n,m)
	   integer n,m
	   real a(:)
	  end subroutine
	 end interface
	 real a(10)
	 real exp(10)
	 data exp /1.,2.,5.,4.,3.,6.,1.,8.,9.,10./
	 do i = 1,10
	  a(i) = i
	 enddo
	 call s( a, 8, 5 )
	 !print *,a
	 call check(a,exp,10)
	end program
