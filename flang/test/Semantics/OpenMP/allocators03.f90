! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.2
! 6.7 allocators construct
! Only the allocate clause is allowed on the allocators construct

subroutine allocate()
use omp_lib

  integer, allocatable :: arr1(:), arr2(:)

  !ERROR: PRIVATE clause is not allowed on the ALLOCATORS directive
  !$omp allocators allocate(arr1) private(arr2)
    allocate(arr1(23), arr2(2))

end subroutine allocate
