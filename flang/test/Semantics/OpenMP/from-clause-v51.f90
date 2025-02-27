!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

subroutine f01(x)
  integer :: x(10)
!ERROR: 'iterator' modifier cannot occur multiple times
  !$omp target update from(iterator(i = 1:5), iterator(j = 1:5): x(i + j))
end

subroutine f03(x)
  integer :: x(10)
!ERROR: 'expectation' modifier cannot occur multiple times
  !$omp target update from(present, present: x)
end

subroutine f04
!ERROR: 'f04' must be a variable
  !$omp target update from(f04)
end
