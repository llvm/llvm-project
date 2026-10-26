! RUN: %flang_fc1 -fdebug-dump-pft %s 2>&1 | FileCheck %s

! Detection of loops whose control flow is structured on the outside but has
! raw branching confined to the body. Such a loop is marked '~' in the dump, as
! opposed to '!' for a fully unstructured one and no marker for a structured
! one.

! Two forward GOTOs that both land inside the body. Nothing enters the body
! from outside and nothing leaves it, so the loop qualifies even though its
! internals are branch-based.
subroutine internal_gotos(a, n)
  real :: a(n)
  ! CHECK:   <<DoConstruct~>> -> 10
  ! CHECK:     6 GotoStmt! -> 8: goto 20
  ! CHECK:     8 ^AssignmentStmt <- 6: 20 a(i) = a(i) + 1.0
  ! CHECK:   <<End DoConstruct~>>
  do i = 1, n
    if (a(i) > 0.0) goto 10
    a(i) = 1.0
    goto 20
10  a(i) = 2.0
20  a(i) = a(i) + 1.0
  end do
end subroutine

! A CYCLE targets the EndDoStmt, which is the boundary between the body and the
! loop control, so it does not count as escaping the body.
subroutine cycle_in_if_block(a, n)
  real :: a(n)
  ! CHECK:   <<DoConstruct~>> -> 8
  ! CHECK:     [[CYC:[0-9]+]] CycleStmt! -> [[END:[0-9]+]]: cycle
  ! CHECK:     [[END]] ^EndDoStmt -> 1 <- [[CYC]]: end do
  ! CHECK:   <<End DoConstruct~>>
  do i = 1, n
    if (a(i) > 0.0) then
      a(i) = 1.0
      cycle
    end if
    a(i) = 2.0
  end do
end subroutine

! The GOTO leaves the loop entirely, so the branch graph is not contained.
subroutine escaping_goto(a, n)
  real :: a(n)
  ! CHECK:   <<DoConstruct!>> -> 7
  ! CHECK:   <<End DoConstruct!>>
  do i = 1, n
    if (a(i) > 0.0) goto 20
    a(i) = 2.0
  end do
20 continue
end subroutine

! A RETURN escapes the loop and the procedure both.
subroutine body_has_return(a, n)
  real :: a(n)
  ! CHECK:   <<DoConstruct!>> -> 7
  ! CHECK:   <<End DoConstruct!>>
  do i = 1, n
    if (a(i) > 0.0) return
    a(i) = 3.0
  end do
end subroutine

! The branch originates outside the loop and lands inside its body, so the body
! has an external entry point and cannot be wrapped.
subroutine incoming_from_outside(a, n)
  real :: a(n)
  ! CHECK:   <<DoConstruct!>> -> 8
  ! CHECK:   <<End DoConstruct!>>
  if (n < 0) goto 30
  do i = 1, n
    a(i) = 4.0
30  continue
  end do
end subroutine

! An inner infinite DO has no structured loop control to preserve.
subroutine infinite_inner(a, n)
  real :: a(n)
  ! CHECK:   <<DoConstruct!>> -> 9
  ! CHECK:     <<DoConstruct!>> -> 8
  ! CHECK:     <<End DoConstruct!>>
  ! CHECK:   <<End DoConstruct!>>
  do i = 1, n
    do
      a(i) = 1.0
      if (a(i) > 0.0) exit
    end do
  end do
end subroutine

! The shape that motivates this work: a forward GOTO raised inside a nested
! IF, jumping over a whole inner DO construct and landing on the last statement
! of the outer loop body. Both endpoints are inside the body, so the outer loop
! is category (c) even though the branch crosses construct boundaries. Note the
! two inner DO constructs stay structured -- only the outer loop carries the
! branching.
subroutine goto_over_inner_loop(qfx, hfx, a, its, ite, jts, jte, force, flux)
  real :: qfx(ite,jte), hfx(ite,jte), a(ite,jte)
  logical :: force
  integer :: flux
  ! CHECK:   <<DoConstruct~>> -> 18
  ! CHECK:     <<DoConstruct>> -> 6
  ! CHECK:     <<End DoConstruct>>
  ! CHECK:       [[GOTO:[0-9]+]] ^GotoStmt! -> [[TGT:[0-9]+]]: goto 350
  ! CHECK:     <<DoConstruct>> -> [[TGT]]
  ! CHECK:     <<End DoConstruct>>
  ! CHECK:     [[TGT]] ^ContinueStmt <- [[GOTO]]: 350 continue
  ! CHECK:     17 EndDoStmt -> 1: enddo
  ! CHECK:   <<End DoConstruct~>>
  do j = jts, jte
    do 330 i = its, ite
      a(i,j) = a(i,j) + 1.0
330 continue
335 continue
    if (force) then
      if (flux .eq. 1) goto 350
    endif
    do i = its, ite
      qfx(i,j) = 0.
      hfx(i,j) = 0.
    enddo
350 continue
  enddo
end subroutine

! No branching at all; detection must leave it unmarked.
subroutine fully_structured(a, n)
  real :: a(n)
  ! CHECK:   <<DoConstruct>> -> 4
  ! CHECK:   <<End DoConstruct>>
  do i = 1, n
    a(i) = 5.0
  end do
end subroutine
