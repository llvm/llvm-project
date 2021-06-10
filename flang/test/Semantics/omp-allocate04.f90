! RUN: %S/test_errors.sh %s %t %flang_fc1 -fopenmp
! REQUIRES: shell
! OpenMP Version 5.0
! 2.11.3 allocate Directive
! Only the allocator clause is allowed on the allocate directive
subroutine allocate()
use omp_lib

  integer :: x, y

  !$omp allocate(x) allocator(omp_default_mem_alloc)

  !ERROR: PRIVATE clause is not allowed on the ALLOCATE directive
  !$omp allocate(y) private(y)
end subroutine allocate
