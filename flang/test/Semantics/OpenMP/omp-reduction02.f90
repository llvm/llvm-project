! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.15.3.6 Reduction Clause
program omp_reduction

  integer :: i
  integer :: k = 10
  integer :: j = 10

  !ERROR: 'k' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp parallel do reduction(+:k), reduction(-:k)
  do i = 1, 10
    k = k + 1
  end do
  !$omp end parallel do

  !ERROR: 'k' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp parallel do reduction(+:k), reduction(-:j), reduction(+:k)
  do i = 1, 10
    k = k + 1
  end do
  !$omp end parallel do

  !ERROR: 'k' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp parallel do reduction(+:j), reduction(-:k), reduction(+:k)
  do i = 1, 10
    k = k + 1
  end do
  !$omp end parallel do

  !ERROR: 'k' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp parallel do reduction(+:j), reduction(-:k), private(k)
  do i = 1, 10
    k = k + 1
  end do
  !$omp end parallel do
end program omp_reduction
