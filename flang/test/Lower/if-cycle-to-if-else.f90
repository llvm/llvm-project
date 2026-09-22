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

! Two IF/CYCLEs in the same DO, both with statements ahead of the CYCLE, so
! both are reshaped and the second nests inside the first's ELSE branch.

subroutine two_cycles_both_leading(n, v)
  integer :: n, i, v(n)

  ! CHECK-LABEL: Subroutine two_cycles_both_leading
  ! CHECK: <<DoConstruct>>
  ! CHECK-NOT: DoConstruct!
  ! CHECK: IfThenStmt{{.*}}: if(i == 1) then
  ! CHECK: AssignmentStmt: v(i) = 0
  ! CHECK: ElseStmt
  ! CHECK: AssignmentStmt: v(i) = 1
  ! CHECK: IfThenStmt{{.*}}: if(i == 2) then
  ! CHECK: AssignmentStmt: v(i) = 3
  ! CHECK: ElseStmt
  ! CHECK: AssignmentStmt: v(i) = 2
  ! CHECK: <<End DoConstruct>>
  ! CHECK-NOT: CycleStmt
  do i = 1, n
     if (i == 1) then
        v(i) = 0
        cycle
     end if
     v(i) = 1
     if (i == 2) then
        v(i) = 3
        cycle
     end if
     v(i) = 2
  end do
end subroutine two_cycles_both_leading

! The reshaped construct is itself moved: the second IF is reshaped first, then
! the first IF is negated and splices the result into the loop body.

subroutine bare_then_leading(n, v)
  integer :: n, i, v(n)

  ! CHECK-LABEL: Subroutine bare_then_leading
  ! CHECK: <<DoConstruct>>
  ! CHECK-NOT: DoConstruct!
  ! CHECK: IfThenStmt [negate]{{.*}}: if(i == 1) then
  ! CHECK: AssignmentStmt: v(i) = 1
  ! CHECK: IfThenStmt{{.*}}: if(i == 2) then
  ! CHECK: AssignmentStmt: v(i) = 3
  ! CHECK: ElseStmt
  ! CHECK: AssignmentStmt: v(i) = 2
  ! CHECK: <<End DoConstruct>>
  ! CHECK-NOT: CycleStmt
  do i = 1, n
     if (i == 1) then
        cycle
     end if
     v(i) = 1
     if (i == 2) then
        v(i) = 3
        cycle
     end if
     v(i) = 2
  end do
end subroutine bare_then_leading

! Nothing follows the second construct, so there is no ELSE branch to move and
! its CYCLE is left alone. The DO stays structured: the branch goes to the
! EndDoStmt and skips nothing.

subroutine two_cycles_trailing(n, v)
  integer :: n, i, v(n)

  ! CHECK-LABEL: Subroutine two_cycles_trailing
  ! CHECK: <<DoConstruct>>
  ! CHECK-NOT: DoConstruct!
  ! CHECK: IfThenStmt{{.*}}: if(i == 1) then
  ! CHECK: AssignmentStmt: v(i) = 0
  ! CHECK: ElseStmt
  ! CHECK: AssignmentStmt: v(i) = 1
  ! CHECK: IfThenStmt{{.*}}: if(i == 2) then
  ! CHECK: CycleStmt
  ! CHECK: <<End DoConstruct>>
  do i = 1, n
     if (i == 1) then
        v(i) = 0
        cycle
     end if
     v(i) = 1
     if (i == 2) then
        cycle
     end if
  end do
end subroutine two_cycles_trailing

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

! A FORMAT statement is not in the lexical chain, so the predecessor of the
! CYCLE is the assignment ahead of it, and that is what falls through to the
! synthesized ElseStmt.

subroutine format_in_then(n, v)
  integer :: n, i, v(n)

  ! CHECK-LABEL: Subroutine format_in_then
  ! CHECK: <<DoConstruct>>
  ! CHECK-NOT: DoConstruct!
  ! CHECK: IfThenStmt
  ! CHECK-NOT: [negate]
  ! CHECK: AssignmentStmt: v(i) = 7
  ! CHECK: FormatStmt
  ! CHECK: ElseStmt
  ! CHECK: AssignmentStmt: v(i) = 2
  ! CHECK: EndIfStmt
  ! CHECK: <<End IfConstruct>>
  ! CHECK: EndDoStmt
  ! CHECK: <<End DoConstruct>>
  ! CHECK-NOT: CycleStmt
  do i = 1, n
     if (v(i) == 1) then
        v(i) = 7
100     format(I5)
        cycle
     end if
     v(i) = 2
  end do
end subroutine format_in_then

! Everything between the construct and the EndDoStmt is a FORMAT, so there is
! no statement for an ELSE branch and the CYCLE is left alone.

subroutine format_only_after_if(n, v)
  integer :: n, i, v(n)

  ! CHECK-LABEL: Subroutine format_only_after_if
  ! CHECK: CycleStmt
  do i = 1, n
     if (v(i) == 1) then
        v(i) = 7
        cycle
     end if
200  format(I5)
  end do
end subroutine format_only_after_if
