!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=45

subroutine f00(x)
  integer :: x(10)
!ERROR: Reference to x must be a contiguous object
  !$omp target update from(x(1:10:2))
end

subroutine f01(x)
  integer :: x(10)
!WARNING: Iterator modifiers are not supported in OpenMP v4.5, try -fopenmp-version=51
  !$omp target update from(iterator(i = 1:5): x(i))
end

subroutine f02(x)
  integer :: x(10)
!WARNING: The PRESENT modifier is not supported in OpenMP v4.5, try -fopenmp-version=51
!WARNING: Iterator modifiers are not supported in OpenMP v4.5, try -fopenmp-version=51
  !$omp target update from(present, iterator(i = 1:5): x(i))
end

subroutine f03(x)
  integer :: x(10)
!WARNING: The PRESENT modifier is not supported in OpenMP v4.5, try -fopenmp-version=51
!ERROR: Only one PRESENT modifier is allowed
  !$omp target update from(present, present: x)
end

subroutine f04
!ERROR: 'f04' must be a variable
  !$omp target update from(f04)
end
