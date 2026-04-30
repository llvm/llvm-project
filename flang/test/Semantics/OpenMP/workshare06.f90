! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! Test that Fortran BLOCK constructs are allowed inside WORKSHARE
! when they contain only WORKSHARE-valid statements.

subroutine test(a, b)
  real :: a, b
  !$omp workshare
  block
    a = b
  end block
  !$omp end workshare
end subroutine
