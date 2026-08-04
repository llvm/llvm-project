! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=60

! The allocator_handle kind is the target's C_INTPTR_T, so a correctly typed
! allocator is accepted without importing omp_lib. This file deliberately does
! not USE omp_lib and so needs no OpenMP runtime modules.

subroutine uses_allocators_without_omp_lib
  use iso_c_binding, only: c_intptr_t
  integer(c_intptr_t) :: my_alloc
  integer :: x

  !$omp target uses_allocators(my_alloc)
  x = 1
  !$omp end target
end subroutine
