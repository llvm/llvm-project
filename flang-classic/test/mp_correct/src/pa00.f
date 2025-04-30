!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Parallel/endparallel directives
*   Lexically nested
	program test
	common/result/result
	integer result(21), expect(21)
	result(1) = 1
	call sub(10)
!	print *, result
	data expect /
     +      1,    2,    6,   12,   20,   30,
     +     42,   56,   72,   90,  110,  108,
     +    104,   98,   90,   80,   68,   54,
     +     38,   20,  100
     + /
	call check(result, expect, 21)
	end
	subroutine sub(n)
!$omp parallel default(shared)
!$omp    do
 	 do i = 1, n
!$omp       parallel shared (i, n)
!$omp          do
               do j = 1, n
		   call work(i,j)
	       enddo
!$omp       endparallel
	 enddo
!$omp endparallel
	end
	subroutine work(i,j)
	common /result/result
	integer result(21)
!$omp critical
	result(i+j) = result(i+j) + i + j
	result(21) = result(21) + 1
!$omp endcritical
	end
