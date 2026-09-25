! RUN: not %flang_fc1 -fsyntax-only -fenumeration-type %s 2>&1 | FileCheck %s
! With the enumeration-type feature enabled, NEXT and PREVIOUS require an
! argument of enumeration type.  A non-enumeration argument must produce a
! proper diagnostic rather than an internal compiler error.

subroutine test_next_nonenum()
  integer :: i, j
  !CHECK: error: Argument of NEXT() must be of enumeration type
  i = next(j)
end subroutine

subroutine test_previous_nonenum()
  integer :: i, j
  !CHECK: error: Argument of PREVIOUS() must be of enumeration type
  i = previous(j)
end subroutine
