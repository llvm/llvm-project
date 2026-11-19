! RUN: not %flang_fc1 -fopenmp -fdebug-unparse %s 2>&1 | FileCheck %s

! With only OpenMP enabled, a directive-looking line between continuation
! lines becomes part of the statement, such as "!$acc" below. This happens
! because blank after the conditional compilation sentinel is optional on
! continuation lines.
! The resulting statement is invalid. gfortran behaves the same.
! With OpenACC enabled, the statement is also invalid, but for another reason:
! the "!$acc" directive should not begin on a continuation line.

! CHECK: error: expected end of statement
subroutine acc_line()
  integer :: x
  x = 1 + &
!$acc loop
      2
end subroutine
