!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=45

! DEPOBJ is not allowed in 4.5, but we should still handle it without crashing.

subroutine f00
  integer :: x
  !ERROR: One of DEPEND, DESTROY or UPDATE (depend-objects) clauses is required on DEPOBJ directive
  !ERROR: The DEPOBJ directive requires a single clause
  !$omp depobj(x)
end
