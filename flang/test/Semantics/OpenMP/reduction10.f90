! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.15.3.6 Reduction Clause
program omp_reduction

  integer :: i
  integer :: k = 10

  !ERROR: Invalid reduction identifier in REDUCTION clause.
  !$omp parallel do reduction(foo:k)
  do i = 1, 10
    k = foo(k)
  end do
  !$omp end parallel do
end program omp_reduction
