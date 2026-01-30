! RUN: %python %S/test_errors.py %s %flang_fc1
! Test that SIZE of TRANSFER with non-constant SOURCE argument
! is not accepted as a constant expression (per F2018 10.1.12)

program test
  implicit none
  type t
    integer :: n
  end type t

  type(t) :: s
  type(t), parameter :: s_const = t(42)

  !ERROR: Must be a constant value
  integer :: k = size(transfer(s, [1]))
  !ERROR: Must be a constant value
  integer, parameter :: m = size(transfer(s, [1]))

  integer :: ok1 = size(transfer(s_const, [1]))
  integer, parameter :: ok2 = size(transfer(s_const, [1]))

end program test
