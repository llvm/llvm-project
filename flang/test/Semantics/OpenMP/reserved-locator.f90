!RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=60

subroutine f
!ERROR: 'OMP_SOME_MEMORY' is not a valid locator
  !$omp target update from(omp_some_memory)
end
