! Check that a loop whose body contains a statically known infinite loop is not
! reclassified. Its structured form would place the body in an
! scf.execute_region with no memory effects, which DCE deletes outright --
! dropping the non-termination and letting execution continue past the loop.
!
! `!` marks a loop left unstructured, `~` one whose branching is confined to
! its body.

! RUN: %flang_fc1 -fdebug-dump-pft -o /dev/null %s 2>&1 | FileCheck %s

! A GO TO branching to itself never leaves the body.
subroutine self_cycle(n)
  integer :: n, i
  do i = 1, n
10   goto 10
  end do
  call never_executed()
end subroutine

! CHECK: Subroutine self_cycle
! CHECK: <<DoConstruct!>>

! The way back to the GO TO is not a branch: control reaches the CONTINUE and
! then continues to the GO TO again. The cycle is found by following control
! from the target, not by looking for another GO TO.
subroutine cycle_through_continue(n)
  integer :: n, i
  do i = 1, n
10  continue
    goto 10
  end do
  call never_executed()
end subroutine

! CHECK: Subroutine cycle_through_continue
! CHECK: <<DoConstruct!>>

! Control: the same GO TO pointing forwards. Control leaves the body through
! the EndDoStmt without passing back through it, so the loop qualifies.
subroutine forward_goto(a, n)
  real :: a(n)
  integer :: n, i
  do i = 1, n
    if (a(i) > 0.0) then
      a(i) = 1.0
      goto 90
    end if
    a(i) = 2.0
90  continue
  end do
end subroutine

! CHECK: Subroutine forward_goto
! CHECK: <<DoConstruct~>>

! Control: a nested loop returns to its own DO statement on every iteration.
! That is the loop's own control rather than a way back to the GO TO, so it
! does not disqualify the loop holding it.
subroutine nested_loop_iterates(a, n)
  real :: a(n,n)
  integer :: n, i, j
  do i = 1, n
    do j = 1, n
      a(i,j) = 0.0
    end do
    if (a(i,1) > 0.0) then
      a(i,1) = 2.0
      goto 90
    end if
    a(i,1) = 1.0
90  continue
  end do
end subroutine

! CHECK: Subroutine nested_loop_iterates
! CHECK: <<DoConstruct~>>
! CHECK: <<DoConstruct>>
! CHECK: <<End DoConstruct>>
! CHECK: <<End DoConstruct~>>

! Two GO TOs branching to each other form the same exit-free cycle.
subroutine mutual_cycle(n)
  integer :: n, i
  do i = 1, n
20   goto 30
30   goto 20
  end do
  call never_executed()
end subroutine

! CHECK: Subroutine mutual_cycle
! CHECK: <<DoConstruct!>>

! Control: nothing branches here at all. The ASSIGN alone makes label 41 a
! branch target, which is what gives the body a block of its own, and with no
! branch there is nothing to trap control. The loop still qualifies.
subroutine label_target_in_body(a, b)
  real :: a(10), b(10)
  integer :: m
  do 42 i = 1, 10
    assign 41 to m
41  a(i) = b(i)
42 continue
end subroutine

! CHECK: Subroutine label_target_in_body
! CHECK: <<DoConstruct~>>

! The way back runs through a construct. Control leaves the CONTINUE, passes
! through the IF, and reaches the GO TO again, so following every successor
! rather than a single path is what finds it. The body holds nothing else, so
! its region would carry no memory effects and DCE would delete it outright.
subroutine cycle_through_construct(n)
  integer :: n, i
  do i = 1, n
10  continue
    if (n > 0) then
    end if
    goto 10
  end do
  call never_executed()
end subroutine

! CHECK: Subroutine cycle_through_construct
! CHECK: <<DoConstruct!>>

! An assigned GO TO closes a cycle like any other branch: its targets are the
! labels ASSIGNed to its variable, which its successors already name.
subroutine assigned_goto_cycle(n)
  integer :: n, i, m
  do i = 1, n
10  continue
    assign 10 to m
    go to m
  end do
  call never_executed()
end subroutine

! CHECK: Subroutine assigned_goto_cycle
! CHECK: <<DoConstruct!>>

! Control: the inner loop's iteration edge is on the path back to the GO TO.
! Following it would report a cycle, although both loops terminate.
!
! The ASSIGN is needed. It keeps the outer loop unstructured, so the outer loop
! is the one analysed. Without it, only the inner loop is analysed, and its
! EndDoStmt is outside its own body, so the search stops before it reaches an
! iteration edge.
subroutine goto_inside_nested_loop(a, n)
  real :: a(n,n)
  integer :: n, i, j, m
  do i = 1, n
    assign 80 to m
    do j = 1, n
      if (a(i,j) > 0.0) then
        a(i,j) = 1.0
        goto 70
      end if
      a(i,j) = 2.0
70    continue
    end do
80  continue
  end do
end subroutine

! CHECK: Subroutine goto_inside_nested_loop
! CHECK: <<DoConstruct~>>
! CHECK: <<DoConstruct~>>
! CHECK: <<End DoConstruct~>>
! CHECK: <<End DoConstruct~>>
