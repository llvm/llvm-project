! RUN: %S/test_errors.sh %s %t %f18
subroutine s
  parameter(a=1.0)
  !ERROR: IMPLICIT NONE statement after PARAMETER statement
  implicit none
end subroutine
