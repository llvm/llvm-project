!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!       OpenMP Library Routines
!       omp_set_num_threads, omp_get_thread_num, omp_in_parallel

	integer num(8), is, n, NPR
	parameter(NPR = 3)
	include 'ompf.h'

	if (omp_in_parallel()) then
	     print *, 'error - should be serial'
	     stop 1
	endif
	call omp_set_num_threads(NPR)
!$omp parallel, private(n)
	n = omp_get_thread_num() + 1
	num(n) = n
!$omp endparallel
	if (omp_in_parallel()) then
	     print *, 'error - should be serial'
	     stop 2
	endif
	is = 0
	do i = 1, NPR
	   is = is + num(i)
	enddo
!	print *, is
	call check(is, 6, 1)
	end
