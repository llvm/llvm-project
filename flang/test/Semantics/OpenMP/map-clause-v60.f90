!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine f00(a)
  integer :: a(*)
  ! No diagnostic expected, assumed-size arrays are allowed on MAP in 6.0.
  !$omp target map(a)
  !$omp end target
end
