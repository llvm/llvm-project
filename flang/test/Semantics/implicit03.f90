! RUN: %S/test_errors.sh %s %t %f18
subroutine s1
  implicit integer(a-z)
  !ERROR: IMPLICIT NONE statement after IMPLICIT statement
  implicit none
end subroutine

subroutine s2
  implicit integer(a-z)
  !ERROR: IMPLICIT NONE(TYPE) after IMPLICIT statement
  implicit none(type)
end subroutine
