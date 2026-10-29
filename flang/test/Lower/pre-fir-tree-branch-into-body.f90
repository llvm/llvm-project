! A loop qualifies only when its body branching is self-contained. An assigned
! GO TO reaches every label ASSIGNed to its variable, so one outside a loop can
! enter its body -- including through an ASSIGN written after the GO TO itself.
!
! `!` marks a loop left unstructured, `~` one whose branching is confined to
! its body.

! RUN: %flang_fc1 -fdebug-dump-pft -o /dev/null %s 2>&1 | FileCheck %s

! The ASSIGN that puts a body label into the variable comes after the assigned
! GO TO. Collecting the ASSIGNed labels before branches are analyzed records
! label 20 as a target, so the branch into the body is seen.
subroutine assigned_goto_reenters(a, n)
  real :: a(n)
  integer :: m, i, n
  assign 10 to m
5 go to m
10 continue
  do i = 1, n
    a(i) = 1.0
    assign 20 to m
20  a(i) = a(i) + 1.0
  end do
  goto 5
end subroutine

! CHECK: Subroutine assigned_goto_reenters
! CHECK: ^AssignedGotoStmt!
! CHECK: <<DoConstruct!>>

! Control: an ASSIGN naming a body label is not itself a branch. With no
! assigned GO TO to use it, nothing can enter the body and the loop qualifies.
subroutine assign_without_goto(a, b)
  real :: a(10), b(10)
  integer :: m
  do 42 i = 1, 10
    assign 41 to m
41  a(i) = b(i)
42 continue
end subroutine

! CHECK: Subroutine assign_without_goto
! CHECK: <<DoConstruct~>>
