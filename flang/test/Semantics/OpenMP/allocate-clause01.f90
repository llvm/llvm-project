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
