! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=51
! OpenMP Version 5.2
! The allocate clause's allocator modifier must be of type allocator_handle
! and the align modifier must be constant, positive integer expression

subroutine allocate()
    use omp_lib

    integer, allocatable :: a, b, c

    !ERROR: The alignment value should be a constant positive integer
    !$omp allocators allocate(allocator(-2), align(-3): b)
        allocate(b)

    !ERROR: The alignment value should be a constant positive integer
    !$omp allocators allocate(align(-4): c)
        allocate(c)
end subroutine

subroutine parallel_allocate(x, y)
    integer :: x, y

    !$omp parallel private(x) allocate(x)
        x = 1
    !$omp end parallel

    !$omp parallel firstprivate(x) allocate(x)
        y = x
    !$omp end parallel

    !$omp parallel private(x, y) allocate(y) allocate(x)
        x = y
    !$omp end parallel
end subroutine

subroutine parallel_allocate_align(x, alignment)
    integer, parameter :: cache_line = 64
    integer :: x, alignment

    !$omp parallel private(x) allocate(align(cache_line): x)
        x = 1
    !$omp end parallel

    !ERROR: The alignment value should be a constant positive integer
    !$omp parallel private(x) allocate(align(0): x)
    !$omp end parallel

    !ERROR: The alignment value should be a constant positive integer
    !$omp parallel private(x) allocate(align(-4): x)
    !$omp end parallel

    !ERROR: The alignment value should be a constant positive integer
    !$omp parallel private(x) allocate(align(alignment): x)
    !$omp end parallel

    !ERROR: The alignment value should be a power of 2
    !$omp parallel private(x) allocate(align(24): x)
    !$omp end parallel
end subroutine
