!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=45 -Werror

subroutine f00(x)
  integer :: x
!WARNING: INOUTSET task dependence type is not supported in OpenMP v4.5, try -fopenmp-version=52
  !$omp task depend(inoutset: x)
  x = x + 1
  !$omp end task
end

subroutine f01(x)
  integer :: x
!WARNING: MUTEXINOUTSET task dependence type is not supported in OpenMP v4.5, try -fopenmp-version=50
  !$omp task depend(mutexinoutset: x)
  x = x + 1
  !$omp end task
end
