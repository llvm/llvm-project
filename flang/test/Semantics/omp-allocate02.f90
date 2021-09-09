! RUN: %python %S/test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.0
! 2.11.3 allocate Directive
! At most one allocator clause can appear on the allocate directive.

subroutine allocate()
use omp_lib
  integer :: x, y
  integer :: a, b
  real, dimension (:,:), allocatable :: darray

  !$omp allocate(x, y) allocator(omp_default_mem_alloc)

  !ERROR: At most one ALLOCATOR clause can appear on the ALLOCATE directive
  !$omp allocate(x, y) allocator(omp_default_mem_alloc) allocator(omp_default_mem_alloc)

  !$omp allocate(x) allocator(omp_default_mem_alloc)
      allocate ( darray(a, b) )

  !ERROR: At most one ALLOCATOR clause can appear on the ALLOCATE directive
  !$omp allocate(x) allocator(omp_default_mem_alloc) allocator(omp_default_mem_alloc)
      allocate ( darray(a, b) )

end subroutine allocate
