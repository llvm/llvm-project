! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! local derived types with allocatable components are not
! recursive-/thread- safe

	integer ii
	integer omp_get_thread_num
	integer results(4)
	integer expect(4)
	data expect /1, 2, 1, 2/
	call omp_set_num_threads(2)
!$omp parallel, private(ii), shared(results)
	ii = omp_get_thread_num()+1
	call sub(ii, results)
!$omp endparallel
	call check(results, expect, 4)
	end
	subroutine sub(n, r)
	integer r(4)
	type aa
	    integer mem
	    real, dimension(:,:),allocatable :: array
	endtype aa
	type bb
	    integer mem
	    real, dimension(:,:),pointer :: array
	endtype bb
	type (aa)xx
	type (bb)yy
	xx%mem = n
	yy%mem = n
!$omp barrier
!	print 99, '3 identical values:',n,xx%mem,yy%mem
!99	format(a, 3i4)
	r(n) = xx%mem
	r(n + 2) = yy%mem
	end
