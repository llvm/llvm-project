! RUN: not %flang_fc1 -fopenacc -fsyntax-only %s 2>&1 | FileCheck %s

! Check that !$acc routine() (empty parentheses) produces a specific error
! pointing at the empty parens, and that the parser recovers so that
! subsequent clauses (e.g. seq) are still parsed — no spurious
! "expected end of OpenACC directive" or "expected declaration construct"
! after recovery.

subroutine sub1()
end subroutine

module m
  !$acc routine() seq
! CHECK: error: empty parentheses in ROUTINE directive; omit parentheses for the unnamed form

  !$acc routine()
! CHECK: error: empty parentheses in ROUTINE directive; omit parentheses for the unnamed form

contains
  subroutine inner()
    !$acc routine() seq
! CHECK: error: empty parentheses in ROUTINE directive; omit parentheses for the unnamed form
! Verify the recovery did not leave "seq" or the () cases unconsumed.
! The range checked by these CHECK-NOTs extends from the third "empty
! parentheses" match above to the first CHECK below (the "expected '('"
! from the malformed-list section), so they cover the entire empty-parens
! section without bleeding into the intentional errors that follow.
! CHECK-NOT: empty parentheses in ROUTINE directive
! CHECK-NOT: expected declaration construct
! CHECK-NOT: expected end of OpenACC directive
  end subroutine
end module

! Malformed name-list cases: verify the "empty parentheses" recovery does not
! fire for inputs it does not apply to.  A name written without parentheses
! and non-empty-but-malformed lists are left unconsumed by the ROUTINE parser
! and produce errors from the surrounding specification-part context instead.
! Each CHECK-NOT below is ranged between the surrounding CHECK lines so it
! tests only the interval between consecutive malformed-case errors.
module m2
contains
  subroutine sub_a()
  end subroutine
  subroutine sub_b()
    !$acc routine sub_a seq
! CHECK: error: expected '('
! CHECK-NOT: empty parentheses in ROUTINE directive
    !$acc routine(sub_a,) seq
! CHECK: error: expected declaration construct
! CHECK-NOT: empty parentheses in ROUTINE directive
    !$acc routine(sub_a sub_b) seq
! CHECK: error: expected declaration construct
! CHECK-NOT: empty parentheses in ROUTINE directive
  end subroutine
end module
