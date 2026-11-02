! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51
subroutine sub
  integer :: x
  !$omp taskwait depend(in: x) nowait
  !ERROR: A NOWAIT clause may only appear on TASKWAIT if a DEPEND clause is present
  !$omp taskwait nowait
end subroutine sub

