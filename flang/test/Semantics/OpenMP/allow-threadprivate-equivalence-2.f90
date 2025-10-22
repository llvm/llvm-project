!RUN: %python %S/../test_errors.py %s %flang -Werror -fopenmp -famd-allow-threadprivate-equivalence

subroutine f
  integer, save :: y
  integer :: x
  !WARNING: Variable 'x' appears a THREADPRIVATE directive and an EQUIVALENCE statement, which does not conform to the OpenMP API specification.
  !$omp threadprivate(x)
  equivalence(x, y)
end

