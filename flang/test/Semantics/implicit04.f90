! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine s
  parameter(a=1.0)
  !ERROR: IMPLICIT NONE statement after PARAMETER statement
  implicit none
end subroutine
