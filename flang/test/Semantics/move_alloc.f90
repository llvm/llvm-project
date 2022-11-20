! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in move_alloc() subroutine calls
program main
  integer, allocatable :: a(:)[:], b(:)[:], c(:)[:], d(:)[:], f(:)
  !ERROR: 'e' is an ALLOCATABLE coarray and must have a deferred coshape
  integer, allocatable :: e(:)[*]
  integer status, coindexed_status[*]
  character(len=1) message, coindexed_message[*]
  integer :: nonAllocatable(10)
  type t
  end type
  class(t), allocatable :: t1
  type(t), allocatable :: t2

  ! standards conforming
  allocate(a(3)[*])
  a = [ 1, 2, 3 ]
  call move_alloc(a, b, status, message)

  allocate(c(3)[*])
  c = [ 1, 2, 3 ]

  !ERROR: too many actual arguments for intrinsic 'move_alloc'
  call move_alloc(a, b, status, message, 1)

  ! standards non-conforming
  !ERROR: 'from' argument to 'move_alloc' may not be a coindexed object
  call move_alloc(c[1], d)

  !ERROR: 'to' argument to 'move_alloc' may not be a coindexed object
  call move_alloc(c, d[1])

  !ERROR: 'stat' argument to 'move_alloc' may not be a coindexed object
  call move_alloc(c, d, coindexed_status[1])

  !ERROR: 'errmsg' argument to 'move_alloc' may not be a coindexed object
  call move_alloc(c, d, status, coindexed_message[1])

  !ERROR: 'errmsg' argument to 'move_alloc' may not be a coindexed object
  call move_alloc(c, d, errmsg=coindexed_message[1])

  !ERROR: 'errmsg' argument to 'move_alloc' may not be a coindexed object
  call move_alloc(c, d, errmsg=coindexed_message[1], stat=status)

  !ERROR: 'stat' argument to 'move_alloc' may not be a coindexed object
  call move_alloc(c, d, stat=coindexed_status[1])

  !ERROR: 'stat' argument to 'move_alloc' may not be a coindexed object
  call move_alloc(c, d, errmsg=message, stat=coindexed_status[1])

  !ERROR: 'from' argument to 'move_alloc' may not be a coindexed object
  !ERROR: 'to' argument to 'move_alloc' may not be a coindexed object
  !ERROR: 'stat' argument to 'move_alloc' may not be a coindexed object
  !ERROR: 'errmsg' argument to 'move_alloc' may not be a coindexed object
  call move_alloc(c[1], d[1], stat=coindexed_status[1], errmsg=coindexed_message[1])

  !ERROR: Argument #1 to MOVE_ALLOC must be allocatable
  call move_alloc(nonAllocatable, f)
  !ERROR: Argument #2 to MOVE_ALLOC must be allocatable
  call move_alloc(f, nonAllocatable)

  !ERROR: When MOVE_ALLOC(FROM=) is polymorphic, TO= must also be polymorphic
  call move_alloc(t1, t2)
  call move_alloc(t2, t1) ! ok

end program main
