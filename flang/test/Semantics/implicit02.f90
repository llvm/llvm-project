! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
subroutine s1
  implicit none
  !ERROR: IMPLICIT statement after IMPLICIT NONE or IMPLICIT NONE(TYPE) statement
  implicit integer(a-z)
end subroutine

subroutine s2
  implicit none(type)
  !ERROR: IMPLICIT statement after IMPLICIT NONE or IMPLICIT NONE(TYPE) statement
  implicit integer(a-z)
end subroutine
