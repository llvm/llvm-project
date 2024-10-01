! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.15.3.6 Reduction Clause
program omp_reduction
  integer :: k
  ! misspelling. Should be "min"
  !ERROR: Invalid reduction operator in REDUCTION clause.
  !$omp parallel reduction(.min.:k)
  !$omp end parallel
end program omp_reduction
