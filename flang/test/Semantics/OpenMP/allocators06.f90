! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.2
! Inherited from 2.11.3 allocate directive
! The allocate directive must appear in the same scope as the declarations of
! each of its list items and must follow all such declarations.

subroutine allocate()
    use omp_lib
    integer, allocatable :: a
contains
    subroutine test()
        !ERROR: List items must be declared in the same scoping unit in which the ALLOCATORS directive appears
        !$omp allocators allocate(omp_default_mem_alloc: a)
            allocate(a)
    end subroutine
end subroutine
