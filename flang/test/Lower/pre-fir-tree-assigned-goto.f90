! RUN: bbc -pft-test -o %t %s | FileCheck %s

! Verify that assigned GO TO records every reachable branch target on the
! source evaluation, so wrappability analyses see any escape from an
! enclosing DO/IF construct.  The dumper prints the first target after
! "->" and any additional targets after commas -- same convention as the
! computed GO TO and arithmetic IF coverage in
! pre-fir-tree-multiway-branch.f90.
!
! Two source forms are exercised:
!   1. `go to v, (l1, l2, ...)` -- targets come from the explicit label list.
!   2. `go to v`                -- targets come from labels previously
!                                  ASSIGN'd to `v` (assignSymbolLabelMap).

! CHECK-LABEL: Subroutine assigned_goto_with_list
subroutine assigned_goto_with_list(j)
  integer :: j
  assign 10 to j
  ! CHECK: AssignedGotoStmt{{.*}} -> {{[0-9]+}}, {{[0-9]+}}:
  go to j, (10, 20)
10 print *, "ten"
20 print *, "twenty"
end subroutine

! CHECK-LABEL: Subroutine assigned_goto_no_list
subroutine assigned_goto_no_list(j)
  integer :: j
  assign 10 to j
  assign 20 to j
  ! CHECK: AssignedGotoStmt{{.*}} -> {{[0-9]+}}, {{[0-9]+}}:
  go to j
10 print *, "ten"
20 print *, "twenty"
end subroutine

! CHECK-LABEL: Subroutine assigned_goto_repeated_label
subroutine assigned_goto_repeated_label(j)
  integer :: j
  assign 10 to j
  ! The label list repeats 10.  Dedup means the source lists exactly two
  ! distinct successors, not three -- the trailing ':' anchors the check so
  ! a third comma-separated target would fail the match.
  ! CHECK: AssignedGotoStmt{{.*}} -> {{[0-9]+}}, {{[0-9]+}}:
  go to j, (10, 10, 20)
10 print *, "ten"
20 print *, "twenty"
end subroutine
