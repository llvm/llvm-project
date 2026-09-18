! RUN: %flang_fc1 -mmlir --wrap-unstructured-constructs-in-execute-region -fdebug-dump-pft %s 2>&1 | FileCheck %s

! An IF body whose last statement is a CYCLE becomes an IF/ELSE, with the
! statements after the construct moved into the ELSE branch. The condition is
! not negated, the statements ahead of the CYCLE must still run when it holds.
! A `!` suffix in this dump marks a construct as unstructured.

subroutine trailing_cycle(n, v)
  integer :: n, i, v(n)

  ! CHECK-LABEL: Subroutine trailing_cycle
  ! CHECK: <<DoConstruct>>
  ! CHECK-NOT: DoConstruct!
  ! CHECK: NonLabelDoStmt{{.*}}: do i = 1, n
  ! CHECK: <<IfConstruct>>
  ! CHECK-NOT: [negate]
  ! CHECK: IfThenStmt{{.*}}: if(i == 1) then
  ! CHECK: AssignmentStmt: v(i) = 0
  ! CHECK: ElseStmt
  ! CHECK: AssignmentStmt: v(i) = 1
  ! CHECK: EndIfStmt
  ! CHECK: <<End IfConstruct>>
  ! CHECK: EndDoStmt
  ! CHECK: <<End DoConstruct>>
  ! CHECK-NOT: CycleStmt
  do i = 1, n
     if (i == 1) then
        v(i) = 0
        cycle
     end if
     v(i) = 1
  end do
end subroutine trailing_cycle

! A CYCLE at the end of an ELSE branch is left alone: it is in the same
! position within the construct, but the rewrite does not apply.

subroutine cycle_in_else(n, v)
  integer :: n, i, v(n)

  ! CHECK-LABEL: Subroutine cycle_in_else
  ! CHECK: CycleStmt
  do i = 1, n
     if (i == 1) then
        v(i) = 0
     else
        v(i) = 1
        cycle
     end if
     v(i) = 2
  end do
end subroutine cycle_in_else

! A CYCLE naming an outer construct is left alone.

subroutine named_outer_cycle(n, v)
  integer :: n, i, j, v(n)

  ! CHECK-LABEL: Subroutine named_outer_cycle
  ! CHECK: CycleStmt
  outer: do i = 1, n
    inner: do j = 1, n
             if (j == 1) then
                v(i) = 0
                cycle outer
             end if
             v(i) = 1
           end do inner
         end do outer
end subroutine named_outer_cycle
