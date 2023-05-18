! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in move_alloc() subroutine calls
program main
  integer, allocatable :: a(:)[:], b(:)[:], f(:), g(:)
  type alloc_component
    integer, allocatable :: a(:)
  end type
  type(alloc_component) :: c[*], d[*]
  !ERROR: 'e' is an ALLOCATABLE coarray and must have a deferred coshape
  integer, allocatable :: e(:)[*]
  integer status, coindexed_status[*]
  character(len=1) message, coindexed_message[*]
  integer :: nonAllocatable(10)
  type t
  end type
  class(t), allocatable :: t1
  type(t), allocatable :: t2
  character, allocatable :: ca*2, cb*3

  ! standards conforming
  allocate(a(3)[*])
  a = [ 1, 2, 3 ]
  call move_alloc(a, b, status, message)

  !ERROR: too many actual arguments for intrinsic 'move_alloc'
  call move_alloc(a, b, status, message, 1)

  ! standards non-conforming
  !ERROR: 'from' argument to 'move_alloc' may not be a coindexed object
  call move_alloc(c[1]%a, f)

  !ERROR: 'to' argument to 'move_alloc' may not be a coindexed object
  call move_alloc(f, d[1]%a)

  !ERROR: 'stat' argument to 'move_alloc' may not be a coindexed object
  call move_alloc(f, g, coindexed_status[1])

  !ERROR: 'errmsg' argument to 'move_alloc' may not be a coindexed object
  call move_alloc(f, g, status, coindexed_message[1])

  !ERROR: 'errmsg' argument to 'move_alloc' may not be a coindexed object
  call move_alloc(f, g, errmsg=coindexed_message[1])

  !ERROR: 'errmsg' argument to 'move_alloc' may not be a coindexed object
  call move_alloc(f, g, errmsg=coindexed_message[1], stat=status)

  !ERROR: 'stat' argument to 'move_alloc' may not be a coindexed object
  call move_alloc(f, g, stat=coindexed_status[1])

  !ERROR: 'stat' argument to 'move_alloc' may not be a coindexed object
  call move_alloc(f, g, errmsg=message, stat=coindexed_status[1])

  !ERROR: 'from' argument to 'move_alloc' may not be a coindexed object
  !ERROR: 'to' argument to 'move_alloc' may not be a coindexed object
  !ERROR: 'stat' argument to 'move_alloc' may not be a coindexed object
  !ERROR: 'errmsg' argument to 'move_alloc' may not be a coindexed object
  call move_alloc(c[1]%a, d[1]%a, stat=coindexed_status[1], errmsg=coindexed_message[1])

  !ERROR: Argument #1 to MOVE_ALLOC must be allocatable
  call move_alloc(nonAllocatable, f)
  !ERROR: Argument #2 to MOVE_ALLOC must be allocatable
  call move_alloc(f, nonAllocatable)

  !ERROR: When MOVE_ALLOC(FROM=) is polymorphic, TO= must also be polymorphic
  call move_alloc(t1, t2)
  call move_alloc(t2, t1) ! ok

  !ERROR: Actual argument for 'to=' has bad type or kind 'CHARACTER(KIND=1,LEN=3_8)'
  call move_alloc(ca, cb)

  !ERROR: Argument #1 to MOVE_ALLOC must be allocatable
  call move_alloc(f(::2), g)
  !ERROR: Argument #2 to MOVE_ALLOC must be allocatable
  call move_alloc(f, g(::2))

end program main
