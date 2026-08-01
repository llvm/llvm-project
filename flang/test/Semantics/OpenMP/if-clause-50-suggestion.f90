!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50

subroutine f
  !OK
  !$omp simd if(.true.)
  do i = 1, 10
  end do

  !ERROR: IF clause is not allowed on TEAMS directive in OpenMP v5.0, try -fopenmp-version=52
  !$omp teams if(.true.)
  !$omp end teams

  !No test for 6.0 because it requires a directive that is not in 5.0
end
