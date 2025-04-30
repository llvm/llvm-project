!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!       OpenMP Library Routines
!       omp_set_num_threads() called after _mp_cdecl() should not
!       have an effect on the number of copies of a threadprivate
!       common block.

	integer result(4), expect(4)
	common/result/result
!	call omp_set_num_threads(2)
	call sub0
!	print *, result

	data expect/2,3,4,5/
	call check(result, expect, 4)
	end
	subroutine sub0
        common /com/ ic1, ic2
!$omp   threadprivate ( /com/ )
	ic1 = 2
	ic2 = 4
	call omp_set_num_threads(2)
	call sub1
	call sub2
	end
	subroutine sub1
        common /com/ ic1, ic2
	integer omp_get_thread_num
!$omp   threadprivate ( /com/ )
!$omp   parallel copyin(/com/)
	ic1 = ic1 + omp_get_thread_num()
	ic2 = ic2 + omp_get_thread_num()
!$omp   endparallel
	end
	subroutine sub2
	integer omp_get_thread_num
	integer result(4)
	common/result/result
        common /com/ ic1, ic2
!$omp   threadprivate ( /com/ )
!$omp   parallel
	result(1+omp_get_thread_num()) = ic1
	result(3+omp_get_thread_num()) = ic2
!$omp   endparallel
	end
