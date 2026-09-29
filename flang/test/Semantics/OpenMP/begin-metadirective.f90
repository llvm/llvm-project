!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

subroutine f00
  !ERROR: A directive in BEGIN METADIRECTIVE should have a corresponding end-directive
  !$omp begin metadirective when(user={condition(.true.)}: taskwait)
  continue
  !$omp end metadirective
end
