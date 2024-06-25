! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

subroutine test_masked()
  integer :: c = 1
  !ERROR: At most one FILTER clause can appear on the MASKED directive
  !$omp masked filter(1) filter(2)
  c = c + 1
  !$omp end masked
  !ERROR: NOWAIT clause is not allowed on the MASKED directive
  !$omp masked nowait 
  c = c + 2
  !$omp end masked
end subroutine
