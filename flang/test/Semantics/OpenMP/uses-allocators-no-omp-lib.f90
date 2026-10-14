! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=60

! A scalar integer allocator is accepted without importing omp_lib. This file
! deliberately uses a kind other than the kind in the compiler's omp_lib.

subroutine uses_allocators_without_omp_lib
  integer(kind=2) :: my_alloc
  integer :: x

  !$omp target uses_allocators(my_alloc)
  x = 1
  !$omp end target
end subroutine
