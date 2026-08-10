! RUN: %python %S/test_errors.py %s %flang_fc1
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
