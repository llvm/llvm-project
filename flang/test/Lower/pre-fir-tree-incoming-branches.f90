! RUN: bbc -pft-test -o %t %s | FileCheck %s

! A branch recorded as "-> target" on its source must appear as "<- source" on
! its target. Each case captures the indices from the outgoing edge and
! matches them on the incoming one, so the two directions are checked to agree
! rather than merely both being present.

! CHECK-LABEL: Subroutine cycle_after_assignment
subroutine cycle_after_assignment(n, v)
  integer :: n, i, v(n)
  do i = 1, n
    if (v(i) == 1) then
      ! A statement ahead of the CYCLE keeps rewriteIfGotos from folding the
      ! branch into a negated condition, so the CycleStmt survives.
      v(i) = 0
      ! CHECK: [[CYC:[0-9]+]] CycleStmt! -> [[END:[0-9]+]]
      cycle
    end if
    v(i) = 2
  ! CHECK: [[END]] ^EndDoStmt -> {{[0-9]+}} <- [[CYC]]
  end do
end subroutine

! CHECK-LABEL: Subroutine two_gotos_one_target
subroutine two_gotos_one_target(n, v)
  integer :: n, i, v(n)
  do i = 1, n
    if (v(i) == 1) then
      v(i) = 7
      ! CHECK: [[G1:[0-9]+]] GotoStmt! -> [[TGT:[0-9]+]]
      goto 60
    end if
    if (v(i) == 2) then
      v(i) = 8
      ! Two distinct sources converge on one target.
      ! CHECK: [[G2:[0-9]+]] GotoStmt! -> [[TGT]]
      goto 60
    end if
    v(i) = 3
  ! CHECK: [[TGT]] ^ContinueStmt <- [[G1]], [[G2]]
60  continue
  end do
end subroutine

! CHECK-LABEL: Subroutine gotos_at_different_depths
subroutine gotos_at_different_depths(n, v)
  integer :: n, i, j, v(n)
  do i = 1, n
    if (v(i) == 1) then
      v(i) = 7
      ! A source in the loop body.
      ! CHECK: [[D1:[0-9]+]] GotoStmt! -> [[LBL:[0-9]+]]
      goto 70
    end if
    do j = 1, n
      if (v(j) == 2) then
        v(j) = 8
        ! A source nested one loop deeper, branching out of the inner DO to
        ! the same label. Sources at different depths must both be recorded.
        ! CHECK: [[D2:[0-9]+]] GotoStmt! -> [[LBL]]
        goto 70
      end if
    end do
    v(i) = 3
  ! CHECK: [[LBL]] ^ContinueStmt <- [[D1]], [[D2]]
70  continue
  end do
end subroutine
