! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags
! OpenMP Version 5.2
! 6.7 allocators construct
! A list item that appears in an allocate clause must appear as
! one of the variables that is allocated by the allocate-stmt in
! the associated allocator structured block.

subroutine allocate()
use omp_lib

  integer, allocatable :: arr1(:), arr2(:, :), arr3(:), arr4(:, :)

  !$omp allocators allocate(arr3)
    allocate(arr3(3), arr4(4, 4))
  !$omp end allocators

  !ERROR: Object 'arr1' in ALLOCATORS directive not found in corresponding ALLOCATE statement
  !$omp allocators allocate(omp_default_mem_alloc: arr1, arr2)
    allocate(arr2(2, 2))

end subroutine allocate
