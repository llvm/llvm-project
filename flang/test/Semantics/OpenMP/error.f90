!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

subroutine f00(x)
!ERROR: The ERROR directive with AT(EXECUTION) cannot appear in the specification part
  !$omp error at(execution) message("Haaa!")
  integer :: x
end

