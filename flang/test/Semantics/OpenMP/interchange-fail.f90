!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine stray_end1
  !ERROR: Misplaced OpenMP end-directive
  !$omp end interchange
end subroutine

subroutine stray_end2
  print *
  !ERROR: Misplaced OpenMP end-directive
  !$omp end interchange
end subroutine

subroutine stray_begin
  !ERROR: This construct should contain a DO-loop or a loop-nest-generating construct
  !$omp interchange permutation(2,1)
end subroutine

