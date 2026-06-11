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
