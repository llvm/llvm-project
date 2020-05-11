! RUN: %S/test_errors.sh %s %t %f18
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
