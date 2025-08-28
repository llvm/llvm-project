!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=45

subroutine foo(x)
  integer :: x(3, *)
  !$omp task depend(in:x(:,5))
  !$omp end task
  !ERROR: Assumed-size array 'x' must have explicit final subscript upper bound value
  !$omp task depend(in:x(5,:))
  !$omp end task
end

