!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=45 -Werror

subroutine f00(x)
  integer :: x(10)
!WARNING: 'iterator' modifier is not supported in OpenMP v4.5, try -fopenmp-version=50
  !$omp task depend(iterator(i = 1:10), in: x(i))
  x = 0
  !$omp end task
end
