! RUN: %flang_fc1 -fopenacc -fdebug-unparse -w %s | FileCheck %s

! Verify that same-kind duplicate variables in OpenACC data-sharing clauses are
! removed by rewrite-parse-tree, so each variable appears at most once when
! unparsed.

subroutine dedup_pair(x, i)
  integer, intent(inout) :: x, i
  !$acc parallel loop private(x, x)
  do i = 1, 10
  end do
end subroutine
! CHECK-LABEL: SUBROUTINE dedup_pair
! CHECK: PRIVATE(x)
! CHECK-NOT: PRIVATE(x,x)
! CHECK-NOT: PRIVATE(x, x)

subroutine dedup_triple(x, i)
  integer, intent(inout) :: x, i
  !$acc parallel loop private(x, x, x)
  do i = 1, 10
  end do
end subroutine
! CHECK-LABEL: SUBROUTINE dedup_triple
! Three occurrences reduce to one.
! CHECK: PRIVATE(x)
! CHECK-NOT: PRIVATE(x,x)
! CHECK-NOT: PRIVATE(x, x)
