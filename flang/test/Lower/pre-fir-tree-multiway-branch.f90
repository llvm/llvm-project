! RUN: bbc -pft-test -o %t %s | FileCheck %s

! Verify that multiway branches (computed GO TO, arithmetic IF) record
! every branch target on the source evaluation, not only the first one.
! The dumper prints the first target after "->" and any additional targets
! after commas. Repeated labels are deduplicated so an ArithmeticIfStmt
! whose two branches share a label lists that label only once as an
! extra successor.

! CHECK-LABEL: Subroutine multi_target_goto
subroutine multi_target_goto(sel)
  integer :: sel
  ! CHECK: ComputedGotoStmt{{.*}} -> {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
  go to (10, 20, 30), sel
10 print *, "one"
20 print *, "two"
30 print *, "three"
end subroutine

! CHECK-LABEL: Subroutine arith_if_three_distinct
subroutine arith_if_three_distinct(x)
  real :: x
  ! CHECK: ArithmeticIfStmt{{.*}} -> {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
  if (x) 10, 20, 30
10 print *, "neg"
20 print *, "zero"
30 print *, "pos"
end subroutine

! CHECK-LABEL: Subroutine arith_if_repeated_label
subroutine arith_if_repeated_label(x)
  real :: x
  ! The two negative/zero branches share label 10. Dedup means the source
  ! lists exactly two distinct successors (10 and 20), not three -- the
  ! trailing ':' anchors the check so a third comma-separated target would
  ! fail the match.
  ! CHECK: ArithmeticIfStmt{{.*}} -> {{[0-9]+}}, {{[0-9]+}}:
  if (x) 10, 10, 20
10 print *, "np"
20 print *, "pos"
end subroutine
