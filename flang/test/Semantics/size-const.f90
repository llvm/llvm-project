! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror -Wno-unused-variable
! Test that SIZE of TRANSFER with non-constant SOURCE argument
! emits a portability warning.

program test
  implicit none
  type t
    integer :: n
  end type t

  type(t) :: s
  type(t), parameter :: s_const = t(42)

  ! These should warn: s is not a constant (portability issue)
  !PORTABILITY: SIZE(TRANSFER(variable,..)) is not portable [-Wportability]
  integer :: k = size(transfer(s, [1]))
  !PORTABILITY: SIZE(TRANSFER(variable,..)) is not portable [-Wportability]
  integer, parameter :: m = size(transfer(s, [1]))

  integer :: ok1 = size(transfer(s_const, [1]))
  integer, parameter :: ok2 = size(transfer(s_const, [1]))

end program test
