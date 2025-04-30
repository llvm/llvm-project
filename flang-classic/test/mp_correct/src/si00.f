!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   SINGLE & MASTER sections
*
	program test
	common /result/result
	integer result(5), expect(5)
	call sub(1)
	data expect /1, 2, 2, 2, 2/
	call check(result, expect, 5)
	end
	subroutine sub(ii)
	common /result/result
	integer result(5), j
	j = ii
!$omp   parallel firstprivate(ii), private(jj)
        jj = ii

!!! jj is already private, can't be firstprivate again
!$omp   single 
        result(1) = result(1) + jj
	call print(0)
!$omp   endsingle nowait

!$omp   master
        j = j + 1
	call print(1)
!$omp   endmaster
!$omp   barrier
!$omp   single
        result(2) = result(2) + j
	call print(2)
!$omp   endsingle nowait

!$omp   single
        result(3) = result(3) + j
	call print(3)
!$omp   endsingle nowait

!$omp   single
        result(4) = result(4) + j
	call print(4)
!$omp   endsingle nowait

!$omp   single
        result(5) = result(5) + j
	call print(5)
!$omp   endsingle
!$omp   endparallel
	end
	subroutine print(n)
	integer omp_get_thread_num
!	print *, 'section:', n, ' thread:', omp_get_thread_num()
	end
!	subroutine check(ir,ix,n)
!	integer ir(n), ix(n)
!	print *, ir
!	print *, ix
!	end
