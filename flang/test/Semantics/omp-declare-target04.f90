! RUN: %python %S/test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.1
! Check OpenMP construct validity for the following directives:
! 2.14.7 Declare Target Directive

program main
  integer, save :: x, y

  !$omp threadprivate(x)

  !ERROR: A THREADPRIVATE variable cannot appear in a DECLARE TARGET directive
  !ERROR: A THREADPRIVATE variable cannot appear in a DECLARE TARGET directive
  !$omp declare target (x, y)

  !$omp threadprivate(y)
end
