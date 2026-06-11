! RUN: %python %S/test_errors.py %s %flang_fc1
! A named constant defined by a PARAMETER statement may be explicitly typed by a
! later type declaration statement under IMPLICIT NONE(TYPE). This extension is
! accepted silently by default (the portability warning appears only with
! -pedantic); this test verifies no diagnostic is emitted by default.
subroutine s1
  implicit none
  parameter(n=4096)
  integer n
  real a(n)
end

! A named constant that is implicitly typed in a module (via an IMPLICIT
! statement, where IMPLICIT NONE(TYPE) is not in effect) must not be flagged
! as a bad forward reference when it is use-associated into an IMPLICIT NONE
! scope.
module m_implicit_test
  implicit character(len=*) (x)
  parameter(xc = 'abc')
end module
subroutine s2
  use m_implicit_test
  implicit none
  if (xc /= 'abc') stop 1
end
