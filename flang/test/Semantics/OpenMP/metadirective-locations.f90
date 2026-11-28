!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine f00
  !ERROR: A METADIRECTIVE in the specification part cannot have executable variants
  !BECAUSE: The executable variant
  !$omp metadirective otherwise(parallel)
  integer :: x
end

subroutine f01
  integer :: x
  continue
  !ERROR: A METADIRECTIVE in the execution part cannot have declarative variants
  !BECAUSE: The declarative variant
  !$omp metadirective otherwise(threadprivate(x))
end
