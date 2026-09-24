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

! The classification of a loop asks for the full target set rather than the
! recorded successors: branch analysis only sees the ASSIGNs that precede the
! GO TO in program order, while the symbol-to-labels map is complete once it
! has finished.

! Both labels ASSIGN'd to m lie in the loop body, so the branching is
! self-contained and the loop keeps its structured form. The ASSIGN of label 10
! follows the GO TO, so the recorded successors name only label 20 -- index 4
! carries no "<-" edge.
! CHECK-LABEL: Subroutine targets_inside
! CHECK: <<DoConstruct~>>
! CHECK: [[GOTO:[0-9]+]] ^AssignedGotoStmt! -> [[L20:[0-9]+]]: go to m
! CHECK: 10 a(i) = 1.0
! CHECK: [[L20]] ^AssignmentStmt <- [[GOTO]]: 20 a(i) = a(i) + 1.0
! CHECK: <<End DoConstruct~>>
subroutine targets_inside(a, n)
  real :: a(n)
  integer :: m
  assign 20 to m
  do i = 1, n
    go to m
10  a(i) = 1.0
20  a(i) = a(i) + 1.0
    assign 10 to m
  end do
end subroutine

! Label 30 is ASSIGN'd to m as well and lies outside the loop, so a branch can
! leave the body and the loop stays unstructured.
! CHECK-LABEL: Subroutine target_outside
! CHECK: <<DoConstruct!>>
! CHECK: <<End DoConstruct!>>
subroutine target_outside(a, n)
  real :: a(n)
  integer :: m
  assign 20 to m
  do i = 1, n
    go to m
10  a(i) = 1.0
    assign 30 to m
  end do
20 continue
30 continue
end subroutine
