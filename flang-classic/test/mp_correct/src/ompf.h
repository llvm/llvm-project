!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!	logical OpenMP functions
	logical omp_get_dynamic, omp_get_nested
	external omp_get_dynamic, omp_get_nested
	logical omp_in_parallel, omp_test_lock
	external omp_in_parallel, omp_test_lock
!	integer OpenMP functions
	integer omp_get_max_threads, omp_get_num_procs
	external omp_get_max_threads, omp_get_num_procs
	integer omp_get_num_threads, omp_get_thread_num
	external omp_get_num_threads, omp_get_thread_num
!	OpenMP subroutines
	external omp_destroy_lock, omp_init_lock
	external omp_set_dynamic, omp_set_lock
	external omp_set_nested, omp_set_num_threads
	external omp_unset_lock
