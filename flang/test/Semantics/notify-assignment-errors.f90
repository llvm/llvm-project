! RUN: %python %S/test_errors.py %s %flang_fc1
! F2023 C933

module notify_defined_assignment
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
  subroutine host_associated_assignment
    use iso_fortran_env, only: notify_type
    type(t), save :: d[*]
    type(notify_type), save :: n[*]
    !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
    d[1, notify=n] = 1
  end subroutine
end module

module notify_type_bound_assignment
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

program invalid_notify_contexts
  use iso_fortran_env, only: notify_type, event_type, team_type
  use notify_defined_assignment
  use notify_type_bound_assignment, only: bound_t => t
  implicit none
  integer, target :: a(4)[*], b[*], local(4)
  integer :: i
  integer, pointer :: q, qa(:)
  type(event_type) :: e[*]
  type(notify_type) :: n[*], na(4)[*]
  type(team_type) :: teams(4)
  character(8) :: text[*]
  type(t) :: d[*]
  type(bound_t) :: bound_d[*]

  a(1)[1, notify=n] = 1

  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  event post(e[1, notify=n])
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  b = a(1)[1, notify=n]
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  call consume(a(1)[1, notify=n])
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  print *, a(1)[1, notify=n]
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  a(1)[1, notify=n] = a(2)[1, notify=n]
  !ERROR: A coindexed object may not be a pointer target
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  q => a(1)[1, notify=n]
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  qa(a(1)[1, notify=n]:) => local
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  d[1, notify=n] = 1
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  bound_d[1, notify=n] = 1
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  where (a(:)[1, notify=n] > 0) a(:)[1] = 1
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  forall (i=1:a(1)[1, notify=n]) a(i)[1] = 1
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  print *, a(1)[1, notify=n]
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  pointer_result(a(1)[1, notify=n]) = 1

  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  a(b[1, notify=n])[2] = 15
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  local(b[1, notify=n]) = 16
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  a(b[1, notify=n])[2, &
    notify=na(1)] = 17
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  text[1](b[1, notify=n]:4) = 'abc'
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  a(1)[b[1, notify=n], &
    notify=n] = 18
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  d[1]%values(b[1, notify=n]) = 18
  a(1)[1, notify=na( &
    !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
    b[1, notify=n])] = 19
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  a(1)[1, stat=local(b[1, notify=n])] = 20
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  a(1)[1, team=teams(b[1, notify=n])] = 21
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  a(1)[1, team_number=b[1, notify=n]] = 22
  ! The inner NOTIFY= violates F2023 C933 independently of F2023 C1178.
  b[1, notify=n[2, &
    !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
    notify=n]] = 23
  !ERROR: coindexed reference has multiple NOTIFY= specifiers
  b[1, notify=n, notify=n] = 24

contains
  subroutine consume(x)
    integer, intent(in) :: x
  end subroutine
  function pointer_result(i) result(p)
    integer, intent(in) :: i
    integer, pointer :: p
    p => local(i)
  end function
end program

subroutine renamed_defined_assignment
  use iso_fortran_env, only: notify_type
  use notify_defined_assignment, only: t, renamed => assign_integer
  interface assignment(=)
    procedure renamed
  end interface
  type(t), save :: d[*]
  type(notify_type), save :: n[*]
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  d[1, notify=n] = 1
end subroutine

subroutine failed_assignment_analysis
  use iso_fortran_env, only: notify_type
  use notify_defined_assignment, only: t
  implicit none
  integer, save :: b[*]
  type(notify_type), save :: n[*]
  type(t), save :: d[*]
  integer :: local(4)

  b[1, notify=n] = 1

  !ERROR: No explicit type declared for 'undeclared'
  b[1, notify=n] = undeclared
  !ERROR: No explicit type declared for 'missing'
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  missing = b[1, notify=n]
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  print *, b[1, notify=n]
  !ERROR: No explicit type declared for 'missing_nested_rhs'
  !ERROR: NOTIFY= may appear only in the image selector of the variable of an intrinsic assignment statement
  local(b[1, notify=n]) = missing_nested_rhs
  !ERROR: No explicit type declared for 'missing_component_rhs'
  d[1, notify=n]%values(2) = missing_component_rhs
end subroutine
