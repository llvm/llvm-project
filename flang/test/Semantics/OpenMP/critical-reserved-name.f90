! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

subroutine f
  !ERROR: Invalid use of a reserved name 'OMP_C'
  !$omp critical(omp_c)
  !ERROR: Invalid use of a reserved name 'OMP_C'
  !$omp end critical(omp_c)
end
