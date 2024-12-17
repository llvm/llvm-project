! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.15.3.6 Reduction Clause
program omp_Reduction
  integer :: i
  integer, parameter :: k = 10
  common /c/ a, b

  !ERROR: Variable 'k' on the REDUCTION clause is not definable
  !BECAUSE: 'k' is not a variable
  !$omp parallel do reduction(+:k)
  do i = 1, 10
    l = k + 1
  end do
  !$omp end parallel do

  !ERROR: Common block names are not allowed in REDUCTION clause
  !$omp parallel do reduction(*:/c/)
  do i = 1, 10
    l = k + 1
  end do
  !$omp end parallel do
end program omp_Reduction
