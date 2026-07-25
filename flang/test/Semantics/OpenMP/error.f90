!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

subroutine f00(x)
!ERROR: The ERROR directive with AT(EXECUTION) cannot appear in the specification part
  !$omp error at(execution) message("Haaa!")
  integer :: x
end

subroutine f01
!ERROR: an error
  !$omp error at(compilation) severity(fatal) message("an error")
end

subroutine f02
!WARNING: a warning
  !$omp error at(compilation) severity(warning) message("a warning")
end

subroutine f03
!ERROR: ERROR
  !$omp error
end

subroutine f04
!ERROR: ERROR
  !$omp error at(compilation) severity(fatal)
end

subroutine f05
!WARNING: WARNING
  !$omp error at(compilation) severity(warning)
end

subroutine f06(n)
  integer :: n
!ERROR: The MESSAGE clause expression must be of type CHARACTER
  !$omp error at(execution) severity(warning) message(n)
end

subroutine f07(c)
  character(len=8) :: c(4)
!ERROR: The MESSAGE clause expression must be scalar
  !$omp error at(execution) severity(warning) message(c)
end

subroutine f08
  character(kind=4, len=8) :: c
!ERROR: The MESSAGE clause expression must be of default character kind
  !$omp error at(execution) severity(warning) message(c)
end

