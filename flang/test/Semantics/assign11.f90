! RUN: %python %S/test_errors.py %s %flang_fc1
! 10.2.1.2p1(1)
program test
  class(*), allocatable :: pa
  class(*), pointer :: pp
  class(*), allocatable :: pac[:]
  type t
    real, allocatable :: a
  end type
  type(t) auc[*]
  pa = 1 ! ok
  !ERROR: Left-hand side of intrinsic assignment may not be polymorphic unless assignment is to an entire allocatable
  pp = 1
  !ERROR: Left-hand side of intrinsic assignment may not be polymorphic if it is a coarray
  pac = 1
  !ERROR: Left-hand side of intrinsic assignment must not be coindexed due to allocatable ultimate component '%a'
  auc[1] = t()
end
