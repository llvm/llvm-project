! REQUIRES: openmp_runtime
! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=50
! OpenMP Version 5.2
! Various checks for the order clause
! 10.3 `order` Clause

! Case 1
subroutine omp_order_runtime_api_call_01()
    use omp_lib
    integer :: i
    !$omp do order(concurrent)
    do i = 1, 5
        !ERROR: The OpenMP runtime API calls are not allowed in the `order(concurrent)` clause region
        print*, omp_get_thread_num()
    end do
    !$omp end do
end subroutine omp_order_runtime_api_call_01

subroutine omp_order_runtime_api_call_02()
    use omp_lib
    integer :: i, num_threads
    !$omp do order(concurrent)
    do i = 1, 5
        !ERROR: The OpenMP runtime API calls are not allowed in the `order(concurrent)` clause region
        call omp_set_num_threads(num_threads)
    end do
    !$omp end do
end subroutine omp_order_runtime_api_call_02

! Case 2
subroutine test_order_threadprivate()
    integer :: i, j = 1, x
    !$omp threadprivate(j)
    !$omp parallel do order(concurrent)
    do i = 1, 5
        !ERROR: A THREADPRIVATE variable cannot appear in an `order(concurrent)` clause region, the behavior is unspecified
        j = x + 1
    end do
    !$omp end parallel do
end subroutine

! Case 3
subroutine omp_order_duplicate_01()
    implicit none
    integer :: i, j
    !ERROR: At most one ORDER clause can appear on the TARGET PARALLEL DO SIMD directive
    !$OMP target parallel do simd ORDER(concurrent) ORDER(concurrent)
    do i = 1, 5
        j = j + 1
    end do
    !$omp end target parallel do simd
end subroutine

subroutine omp_order_duplicate_02()
    integer :: i, j
    !$omp teams
    !ERROR: At most one ORDER clause can appear on the DISTRIBUTE PARALLEL DO SIMD directive
    !$omp distribute parallel do simd order(concurrent) order(concurrent)
    do i = 1, 5
        j = j + 1
    end do
    !$omp end distribute parallel do simd
    !$omp end teams
end subroutine
