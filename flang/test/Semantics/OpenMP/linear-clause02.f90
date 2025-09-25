!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

subroutine f00(x)
  integer :: x
  !WARNING: The 'modifier(<list>)' syntax is deprecated in OpenMP v5.2, use '<list> : modifier' instead
  !$omp declare simd linear(uval(x))
end

subroutine f01(x)
  integer :: x
  !ERROR: An exclusive 'step-simple-modifier' modifier cannot be specified together with a modifier of a different type
  !$omp declare simd linear(uval(x) : 2)
end
