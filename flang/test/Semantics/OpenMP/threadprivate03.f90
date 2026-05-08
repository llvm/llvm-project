! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -pedantic
! OpenMP Version 5.1
! Check OpenMP construct validity for the following directives:
! 2.21.2 Threadprivate Directive

module mod1
end

program main
  use mod1
  integer, parameter :: i = 1

  !ERROR: The module name cannot be in a THREADPRIVATE directive
  !$omp threadprivate(mod1)

  ! This is now allowed, since "main" is implicitly declared symbol,
  ! separate from the main program symbol.
  !$omp threadprivate(main)

  !ERROR: The entity with PARAMETER attribute cannot be in a THREADPRIVATE directive
  !$omp threadprivate(i)

contains
  subroutine sub()
    !ERROR: The procedure name cannot be in a THREADPRIVATE directive
    !$omp threadprivate(sub)
  end
end
