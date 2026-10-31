! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s

! Verify the location and number of parser diagnostics for declarations.

! The nonstandard ',' in place of '::' fallback must not add a diagnostic
! unless a ',' follows the attribute list.
subroutine s1
  integer :: d
! CHECK-NOT: error: expected entity declarations
! CHECK: :[[@LINE+2]]:24: error: expected '::'
! CHECK-NOT: error: expected entity declarations
  real, dimension (0:d)[0:d] :: ra
end subroutine

subroutine s2
! CHECK: :[[@LINE+2]]:21: error: expected '::'
! CHECK-NOT: error: expected entity declarations
  real, dimension(3)
end subroutine

! A ',' follows the attribute list, so the fallback still diagnoses the
! bad entity declaration.
subroutine s3
! CHECK: :[[@LINE+2]]:23: error: expected one of
! CHECK-NOT: error: expected '::'
  real, dimension(3), 1x
end subroutine

! The message points at the start of the offending statement.
block data b
  common /c/ x
! CHECK: :[[@LINE+1]]:3: error: expected declaration construct
  continue
end block data
