!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=45

subroutine f00(x)
  integer :: x(10)
!ERROR: THREADSET clause is not allowed on directive TASK in OpenMP v4.5, try -fopenmp-version=60
!$omp task threadset(omp_pool)
  x = x + 1
!$omp end task
end
