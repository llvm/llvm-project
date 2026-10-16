! RUN: %python %S/../test_errors.py %s %flang -fopenacc

module acc_routine_bad

contains
!ERROR: ROUTINE directive without name must appear within the specification part of a subroutine or function definition, or within an interface body for a subroutine or function in an interface block
!$acc routine
subroutine acc_routine20()
end subroutine
end module
