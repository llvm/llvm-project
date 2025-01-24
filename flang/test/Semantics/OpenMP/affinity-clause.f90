!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=45

subroutine f00(x)
  integer :: x(10)
!ERROR: AFFINITY clause is not allowed on directive TASK in OpenMP v4.5, try -fopenmp-version=50
!$omp task affinity(x)
  x = x + 1
!$omp end task
end
