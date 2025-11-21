! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=51
! OpenMP Version 5.1
! 2.11.3 allocate Directive
! If list items within the ALLOCATE directive have the SAVE attribute, are a
! common block name, then only predefined memory allocator parameters can be
! used in the allocator clause

module AllocateModule
  INTEGER :: z
end module

subroutine allocate(custom_allocator)
use omp_lib
use AllocateModule
  integer, SAVE :: x
  integer :: w
  COMMON /CommonName/ y

  integer(kind=omp_allocator_handle_kind) :: custom_allocator

  !$omp allocate(x) allocator(omp_default_mem_alloc)
  !ERROR: A variable that is part of a common block may not be specified as a list item in an ALLOCATE directive, except implicitly via the named common block
  !$omp allocate(y) allocator(omp_default_mem_alloc)
  !ERROR: A list item on a declarative ALLOCATE must be declared in the same scope in which the directive appears
  !$omp allocate(z) allocator(omp_default_mem_alloc)

  !ERROR: If a list item is a named common block or has SAVE attribute, an ALLOCATOR clause must be present with a predefined allocator
  !$omp allocate(x)
  !ERROR: A variable that is part of a common block may not be specified as a list item in an ALLOCATE directive, except implicitly via the named common block
  !$omp allocate(y)
  !ERROR: A list item on a declarative ALLOCATE must be declared in the same scope in which the directive appears
  !$omp allocate(z)

  !$omp allocate(w) allocator(custom_allocator)

  !$omp allocate(x) allocator(custom_allocator)
  !ERROR: A variable that is part of a common block may not be specified as a list item in an ALLOCATE directive, except implicitly via the named common block
  !$omp allocate(y) allocator(custom_allocator)
  !ERROR: A list item on a declarative ALLOCATE must be declared in the same scope in which the directive appears
  !$omp allocate(z) allocator(custom_allocator)
end subroutine allocate
