! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
subroutine s
  parameter(a=1.0)
  !ERROR: IMPLICIT NONE statement after PARAMETER statement
  implicit none
end subroutine
