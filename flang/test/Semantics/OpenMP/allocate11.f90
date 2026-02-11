! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=50
! OpenMP Version 5.0
! 2.11.3 allocate Directive
! If list items within the ALLOCATE directive have the SAVE attribute, are a
! common block name, or are declared in the scope of a module, then only
! predefined memory allocator parameters can be used in the allocator clause

module AllocateModule
  INTEGER :: z
  !ERROR: If a list item is a named common block, has SAVE attribute or is declared in the scope of a module, an ALLOCATOR clause must be present with a predefined allocator
  !$omp allocate(z)
end module

subroutine allocate(custom_allocator)
use omp_lib
use AllocateModule
  integer, SAVE :: x
  integer :: w
  COMMON /CommonName/ y

  integer(kind=omp_allocator_handle_kind) :: custom_allocator

  !ERROR: If a list item is a named common block, has SAVE attribute or is declared in the scope of a module, an ALLOCATOR clause must be present with a predefined allocator
  !$omp allocate(x)
end subroutine allocate
