! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.0
! 2.11.3 allocate Directive
! allocate directives that appear in a target region must specify an allocator
! clause unless a requires directive with the dynamic_allocators clause is present
! in the same compilation unit.

subroutine allocate()
use omp_lib
  integer :: a, b
  real, dimension (:,:), allocatable :: darray

  !$omp target
      !$omp allocate allocator(omp_default_mem_alloc)
          allocate ( darray(a, b) )
  !$omp end target

  !$omp target
      !ERROR: ALLOCATE directives that appear in a TARGET region must specify an allocator clause
      !$omp allocate
          allocate ( darray(a, b) )
  !$omp end target

end subroutine allocate
