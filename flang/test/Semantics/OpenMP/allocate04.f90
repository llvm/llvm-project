! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags
! OpenMP Version 5.0
! 2.11.3 allocate Directive
! Only the allocator clause is allowed on the allocate directive
! List item in ALLOCATE directive must not be a dummy argument
! List item in ALLOCATE directive must not have POINTER attribute
! List item in ALLOCATE directive must not be a associate name
subroutine allocate(z)
use omp_lib
use iso_c_binding

  type(c_ptr), pointer :: p
  integer :: x, y, z

  associate (a => x)
  !$omp allocate(x) allocator(omp_default_mem_alloc)

  !ERROR: PRIVATE clause is not allowed on the ALLOCATE directive
  !$omp allocate(y) private(y)
  !ERROR: List item 'z' in ALLOCATE directive must not be a dummy argument
  !$omp allocate(z)
  !ERROR: List item 'p' in ALLOCATE directive must not have POINTER attribute
  !$omp allocate(p)
  !ERROR: List item 'a' in ALLOCATE directive must not be an associate name
  !$omp allocate(a)
  end associate
end subroutine allocate
