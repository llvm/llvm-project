!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 
!       array expression in parallel loops
        program tomp
        implicit none

        integer n
        parameter(n=100)
        real A(n,n)
        integer i,j
        real dsum
	real result(2),expect(2)
	data expect/1020000,1020000/

        do i=1,n
        do j=1,n
           A(i,j) = real(i+j+1)
        enddo
        enddo


        dsum = 0.0
!$omp   parallel do private(i) reduction(+:dsum)
        do i=1,n
          dsum = dsum + sum(A(i,:n))  !a$r is SC_LOCAL, but should be private
        enddo

	result(1) = dsum
        !print*,' dsum in i', dsum


        dsum = 0.0
!$omp   parallel do private(j) reduction(+:dsum)
        do j=1,n
          dsum = dsum + sum(A(:n, j)) !a$r1 is SC_LOCAL, but should be private
        enddo

        !print*,'dsum in j ', dsum
	result(2) = dsum

        call check(result,expect,2)
        end

