!RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=60

subroutine f
!ERROR: Invalid use of a reserved name 'OMP_SOME_MEMORY'
  !$omp target update from(omp_some_memory)
end
