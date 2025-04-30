!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!	OpenMP Parallel Region
!	parallel private subroutine call

	program p
	parameter(n=10)
	integer result(n)
	integer expect(n)
	data expect/101,201,301,401,102,202,302,402,103,203/
	call sp1(result,n)
	!print *,result
	call check(result,expect,n)
	end

	subroutine sp1(x,n)
	 integer n
	 integer x(n)
	 integer omp_get_thread_num
	 integer omp_get_num_threads
!$omp   parallel private(iam,np,ipoints)
	 iam = omp_get_thread_num()+1
	 np = omp_get_num_threads()
	 call subdomain(x,iam,n,np)
!$omp   end parallel
	end

	subroutine subdomain(x,iam,n,np)
	integer n
	integer x(n),iam,np
	integer i,j
	j = 0
	do i = iam,n,np
	 j = j + 1
	 x(i) = iam*100 + j
	enddo
	end
