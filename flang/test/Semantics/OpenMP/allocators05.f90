! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=50
! OpenMP Version 5.2
! Inherited from 2.11.3 allocate directive
! allocate directives that appear in a target region must specify an
! allocator clause unless a requires directive with the dynamic_allocators
! clause is present in the same compilation unit.

subroutine allocate()
    use omp_lib

    integer :: i
    integer, allocatable :: a(:), b(:)
    integer, parameter :: LEN = 2

    !$omp target private(a, b)
    !$omp allocators allocate(omp_default_mem_alloc: a)
        allocate(a(LEN))
    !ERROR: An ALLOCATE clause in a TARGET region must specify an allocator or REQUIRES(DYNAMIC_ALLOCATORS) must be specified
    !$omp allocators allocate(b)
        allocate(b(LEN))
    !$omp end target
end subroutine
