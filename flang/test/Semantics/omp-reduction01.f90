! RUN: %S/test_errors.sh %s %t %flang_fc1 -fopenmp
! REQUIRES: shell
! OpenMP Version 4.5
! 2.15.3.6 Reduction Clause
program omp_reduction
  integer :: i
  integer :: k = 10

  !ERROR: Invalid reduction operator in REDUCTION clause.
  !$omp parallel do reduction(**:k)
  do i = 1, 10
    k = k ** 1
  end do
  !$omp end parallel do
end program omp_reduction
