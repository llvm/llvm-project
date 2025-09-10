! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -pedantic
! OpenMP Version 5.1
! Check OpenMP construct validity for the following directives:
! 2.14.7 Declare Target Directive

module mod1
end

subroutine bar
  !$omp declare target (bar)
end subroutine

program main
  use mod1

  !ERROR: The module name cannot be in a DECLARE TARGET directive
  !$omp declare target (mod1)

  ! This is now allowed: "main" is implicitly declared symbol separate
  ! from the main program symbol
  !$omp declare target (main)
end
