! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.0
! 2.11.3 allocate Directive
! If list items within the ALLOCATE directive have the SAVE attribute, are a common block name, or are declared in the scope of a
! module, then only predefined memory allocator parameters can be used in the allocator clause

module AllocateModule
  INTEGER :: z
end module

subroutine allocate()
use omp_lib
use AllocateModule
  integer, SAVE :: x
  integer :: w
  COMMON /CommonName/ y

  integer(kind=omp_allocator_handle_kind) :: custom_allocator
  integer(kind=omp_memspace_handle_kind) :: memspace
  type(omp_alloctrait), dimension(1) :: trait
  memspace = omp_default_mem_space
  trait(1)%key = fallback
  trait(1)%value = default_mem_fb
  custom_allocator = omp_init_allocator(memspace, 1, trait)

  !$omp allocate(x) allocator(omp_default_mem_alloc)
  !$omp allocate(y) allocator(omp_default_mem_alloc)
  !$omp allocate(z) allocator(omp_default_mem_alloc)

  !$omp allocate(x)
  !$omp allocate(y)
  !$omp allocate(z)

  !$omp allocate(w) allocator(custom_allocator)

  !ERROR: If list items within the ALLOCATE directive have the SAVE attribute, are a common block name, or are declared in the scope of a module, then only predefined memory allocator parameters can be used in the allocator clause
  !$omp allocate(x) allocator(custom_allocator)
  !ERROR: If list items within the ALLOCATE directive have the SAVE attribute, are a common block name, or are declared in the scope of a module, then only predefined memory allocator parameters can be used in the allocator clause
  !$omp allocate(y) allocator(custom_allocator)
  !ERROR: If list items within the ALLOCATE directive have the SAVE attribute, are a common block name, or are declared in the scope of a module, then only predefined memory allocator parameters can be used in the allocator clause
  !$omp allocate(z) allocator(custom_allocator)
end subroutine allocate
