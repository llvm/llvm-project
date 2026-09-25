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
