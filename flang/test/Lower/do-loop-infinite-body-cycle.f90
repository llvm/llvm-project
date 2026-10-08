! Check that a loop whose body contains a statically known infinite loop is not
! reclassified. Its structured form would place the body in an
! scf.execute_region with no memory effects, which DCE deletes outright --
! dropping the non-termination and letting execution continue past the loop.
!
! `!` marks a loop left unstructured, `~` one whose branching is confined to
! its body.

! RUN: %flang_fc1 -fdebug-dump-pft -o /dev/null %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s --check-prefix=FIR

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

! The cycle survives as a block branching to itself. No fir.do_loop is emitted:
! the loop control is unstructured too.
! FIR-LABEL: func.func @_QPself_cycle
! FIR:         cf.cond_br %{{.*}}, ^[[SELF:bb[0-9]+]], ^bb{{[0-9]+}}
! FIR:       ^[[SELF]]:
! FIR:         cf.br ^[[SELF]]
! FIR-NOT:     fir.do_loop

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

! Two blocks branching to each other, neither leaving the cycle.
! FIR-LABEL: func.func @_QPmutual_cycle
! FIR:       ^[[A:bb[0-9]+]]:
! FIR:         cf.br ^[[B:bb[0-9]+]]
! FIR:       ^[[B]]:
! FIR:         cf.br ^[[A]]
! FIR-NOT:     fir.do_loop

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

! The control case keeps its structured form, body folded into a region.
! FIR-LABEL: func.func @_QPlabel_target_in_body
! FIR:         fir.do_loop
! FIR:           scf.execute_region no_inline {

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

! A backward GO TO whose cycle control can leave for the end of the body. The
! IF falls through to the rest of the body, so the region's yield stays
! reachable and the loop qualifies; whether the cycle is left is up to the
! program, as for a DO WHILE.
subroutine escapable_backward_goto(a, b, n)
  real :: a(n), b(n), s
  integer :: n, i
  do i = 1, n
    s = a(i)
10  s = s * 0.9
    if (s > 0.1) goto 10
    b(i) = s
  end do
end subroutine

! CHECK: Subroutine escapable_backward_goto
! CHECK: <<DoConstruct~>>

! The loop keeps its structured form, with the backward branch inside the
! region holding the body.
! FIR-LABEL: func.func @_QPescapable_backward_goto
! FIR:         fir.do_loop
! FIR:           scf.execute_region no_inline {
! FIR:           ^[[TGT:bb[0-9]+]]:  // 2 preds
! FIR:             cf.cond_br %{{.*}}, ^[[GOTO:bb[0-9]+]], ^[[REST:bb[0-9]+]]
! FIR:           ^[[GOTO]]:
! FIR-NEXT:        cf.br ^[[TGT]]
! FIR:           ^[[REST]]:
! FIR:             scf.yield

! An escapable cycle next to one that cannot be left. Each GO TO is checked on
! its own, so the exit-free cycle still disqualifies the loop.
subroutine escapable_and_exit_free_cycles(a, b, n)
  real :: a(n), b(n), s
  integer :: n, i
  do i = 1, n
    s = a(i)
10  s = s * 0.9
    if (s > 0.1) goto 10
    if (s < 0.0) then
20    goto 20
    end if
    b(i) = s
  end do
end subroutine

! CHECK: Subroutine escapable_and_exit_free_cycles
! CHECK: <<DoConstruct!>>
