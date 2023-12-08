! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine s1
  implicit none
  !ERROR: More than one IMPLICIT NONE statement
  implicit none(type)
end subroutine

subroutine s2
  implicit none(external)
  !ERROR: More than one IMPLICIT NONE statement
  implicit none
end subroutine
