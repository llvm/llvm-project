! NOTE: Do not autogenerate
! RUN: %flang_fc1 -cpp -fsyntax-only %s
! RUN: %flang_fc1 -cpp -fdebug-dump-parse-tree %s | FileCheck %s

module notify_intrinsic_types
  type t
    integer :: value
    integer :: values(4)
  end type
  interface assignment(=)
    module procedure assign_integer
  end interface
contains
  subroutine assign_integer(lhs, rhs)
    type(t), intent(out) :: lhs
    integer, intent(in) :: rhs
    lhs%value = rhs
  end subroutine
end module

module notify_intrinsic_bound_types
  type t
    integer :: value
  contains
    procedure :: assign_integer
    generic :: assignment(=) => assign_integer
  end type
contains
  subroutine assign_integer(lhs, rhs)
    class(t), intent(out) :: lhs
    integer, intent(in) :: rhs
    lhs%value = rhs
  end subroutine
end module

#define REMOTE b[1, notify=n]

program intrinsic_notify_assignment
  use iso_fortran_env, only: notify_type, team_type
  use notify_intrinsic_types
  use notify_intrinsic_bound_types, only: bound_t => t
  implicit none
  type container
    integer, allocatable :: c[:]
  end type
  integer :: a(4)[*], b[*], local_array(4), status, i
  logical :: mask(4)
  type(t) :: d[*], rhs
  type(bound_t) :: bound_d[*], bound_rhs
  type(notify_type) :: n[*], na(4)[*]
  type(team_type) :: team
  type(container) :: x
  character(8) :: text[*]
  complex :: z[*]

  !CHECK: AssignmentStmt = 'b[1_8,NOTIFY=n]=1_4'
  b[1, notify=n] = 1
  !CHECK: AssignmentStmt = 'a(1_8)[1_8,NOTIFY=na(2_8)]=2_4'
  a(1)[1, notify=na(1+1)] = 2
  a(:)[1, notify=n] = 3
  d[1, notify=n]%value = 4
  text[1, notify=n](2:4) = 'abc'
  z[1, notify=n]%re = 5.
  !CHECK: AssignmentStmt = 'b[1_8,STAT=status,NOTIFY=n]=6_4'
  b[1, notify=n, stat=status] = 6
  !CHECK: AssignmentStmt = 'b[1_8,STAT=status,NOTIFY=n,TEAM=team]=7_4'
  b[1, stat=status, notify=n, team=team] = 7
  !CHECK: AssignmentStmt = 'b[1_8,STAT=status,NOTIFY=n,TEAM_NUMBER=1_4]=8_4'
  b[1, team_number=1, notify=n, stat=status] = 8
  if (.true.) b[1, notify=n] = 9
  where (mask) a(:)[1, notify=n] = 10
  where (mask)
    a(:)[1, notify=n] = 11
  elsewhere
    a(:)[1, notify=n] = 12
  end where
  forall (i=1:4) a(i)[1, notify=n] = 13
  forall (i=1:4)
    a(i)[1, notify=n] = 14
  end forall

  ! A visible but nonmatching generic leaves these assignments intrinsic.
  d[1, notify=n] = rhs
  bound_d[1, notify=n] = bound_rhs

  ! Ordinary coindexed reads in target expressions remain permitted.
  a(b[1])[2, notify=n] = 15
  local_array(b[1]) = 16
  a(1)[b[1], notify=na(1)] = 17
  text[1, notify=n](b[1]:4) = 'abc'
  b[1, &
    notify=n] = 18
  REMOTE = 19
  a(1)[1, notify=na(b[1])] = 20
  allocate(x%c[*])
  x%c[1, notify=n] = 21
  associate (remote => b)
    remote[1, notify=n] = 22
  end associate
  d[1, notify=n]%values(2) = 23
end program
