!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   SECTIONS/ENDSECTIONS
*   Lexically enclosed within a parallel region

	program test
	common /result/result
	integer result(5), expect(5)
	call sub
!	print *, result
!	print *, expect
	data expect /1, 1, 1, 1, 1/
	call check(result, expect, 5)
	end
	subroutine sub
	common /result/result
	integer result(5)
!$omp   parallel
!$omp   sections
        result(1) = result(1) + 1
	call print(0)
!$omp   section
        result(2) = result(2) + 1
	call print(1)
!$omp   section
        result(3) = result(3) + 1
	call print(2)
!$omp   section
        result(4) = result(4) + 1
	call print(3)
!$omp   section
        result(5) = result(5) + 1
	call print(4)
!$omp   endsections
!$omp   endparallel
	end
	subroutine print(n)
	integer omp_get_thread_num
!	print *, 'section:', n, ' thread:', omp_get_thread_num()
	end
