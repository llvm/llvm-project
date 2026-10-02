! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s

! Verify where withMessage() anchors its message when no tokens matched.

! The nonstandard ',' in place of '::' fallback must not outrank the standard
! "expected '::'" diagnostic.
subroutine s
  integer :: d
! CHECK-NOT: error: expected entity declarations
! CHECK: :[[@LINE+2]]:24: error: expected '::'
! CHECK: :[[@LINE+1]]:25: error: expected entity declarations
  real, dimension (0:d)[0:d] :: ra
end subroutine

! The message points at the start of the offending statement.
block data b
  common /c/ x
! CHECK: :[[@LINE+1]]:3: error: expected declaration construct
  continue
end block data
