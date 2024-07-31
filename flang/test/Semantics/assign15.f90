! RUN: %python %S/test_errors.py %s %flang_fc1
! Test error location when assignment starts with macro expansion.

#define X_VAR x
program main
  real(4) :: x
  character(10) :: c
  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types REAL(4) and CHARACTER(KIND=1)
  X_VAR = c
  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types CHARACTER(KIND=1) and REAL(4)
  c = X_VAR
end
