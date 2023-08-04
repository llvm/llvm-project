! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.2
! The allocate clause's allocator modifier must be of type allocator_handle
! and the align modifier must be constant, positive integer expression

subroutine allocate()
    use omp_lib

    integer, allocatable :: a, b, c

    !ERROR: The parameter of the ALLOCATE clause must be a positive integer expression
    !$omp allocators allocate(-1: a)
        allocate(a)

    !ERROR: The parameter of the ALLOCATE clause must be a positive integer expression
    !$omp allocators allocate(allocator(-2), align(-3): b)
        allocate(b)

    !ERROR: The parameter of the ALLOCATE clause must be a positive integer expression
    !$omp allocators allocate(align(-4): c)
        allocate(c)
end subroutine
