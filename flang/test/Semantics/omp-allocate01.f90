! RUN: %S/test_errors.sh %s %t %flang_fc1 -fopenmp
! REQUIRES: shell
! OpenMP Version 5.0
! 2.11.3 allocate Directive
! The allocate directive must appear in the same scope as the declarations of
! each of its list items and must follow all such declarations.

subroutine allocate()
use omp_lib
  integer :: x
  contains
    subroutine sema()
    integer :: a, b
    real, dimension (:,:), allocatable :: darray

    !ERROR: List items must be declared in the same scoping unit in which the ALLOCATE directive appears
    !$omp allocate(x)
        print *, a

    !ERROR: List items must be declared in the same scoping unit in which the ALLOCATE directive appears
    !$omp allocate(x) allocator(omp_default_mem_alloc)
      allocate ( darray(a, b) )
    end subroutine sema

end subroutine allocate
