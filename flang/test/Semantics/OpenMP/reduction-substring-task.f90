! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=50
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=51
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=60

subroutine task_reductions(c)
  character(4) :: c
  !$omp declare reduction(+:character(2),character(4):omp_out=omp_in)
  !ERROR: A substring cannot appear in a TASK_REDUCTION clause
  !$omp taskgroup task_reduction(+:c(2:3))
  !$omp end taskgroup

  !$omp taskgroup task_reduction(+:c)
  !ERROR: A substring cannot appear in a IN_REDUCTION clause
  !$omp task in_reduction(+:c(2:3))
  !$omp end task
  !$omp end taskgroup
end subroutine
