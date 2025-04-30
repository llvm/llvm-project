! @@name:	nthrs_nesting.1f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
	program icv
	use omp_lib
	call omp_set_nested(.true.)
	call omp_set_dynamic(.false.)
!$omp parallel
!$omp parallel
!$omp single
	! If OMP_NUM_THREADS=2,3 was set, the following should print:
	! Inner: num_thds= 3
	! Inner: num_thds= 3
	! If nesting is not supported, the following should print:
	! Inner: num_thds= 1
	! Inner: num_thds= 1
	print *, "Inner: num_thds=", omp_get_num_threads()
!$omp end single
!$omp end parallel
!$omp barrier
	call omp_set_nested(.false.)
!$omp parallel
!$omp single
	! Even if OMP_NUM_THREADS=2,3 was set, the following should print,
	! because nesting is disabled:
	! Inner: num_thds= 1
	! Inner: num_thds= 1
	print *, "Inner: num_thds=", omp_get_num_threads()
!$omp end single
!$omp end parallel
!$omp barrier
!$omp single
	! If OMP_NUM_THREADS=2,3 was set, the following should print:
	! Outer: num_thds= 2
	print *, "Outer: num_thds=", omp_get_num_threads()
!$omp end single
!$omp end parallel
	end
