! RUN: %B/test/Semantics/test_errors.sh %s %flang %t
subroutine s
  parameter(a=1.0)
  !ERROR: IMPLICIT NONE statement after PARAMETER statement
  implicit none
end subroutine
