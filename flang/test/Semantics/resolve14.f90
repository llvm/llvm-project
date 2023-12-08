! RUN: %python %S/test_errors.py %s %flang_fc1
module m1
  integer :: x
  integer :: y
  integer :: z
  integer, parameter :: k1 = selected_int_kind(9)
end
module m2
  real :: y
  real :: z
  real :: w
  integer, parameter :: k2 = selected_int_kind(9)
end

subroutine p1
  use m1
  use m2
  ! check that selected_int_kind is not use-associated
  integer, parameter :: k = selected_int_kind(9)
end

program p2
  use m1, xx => x, y => z
  use m2
  volatile w
  !ERROR: Cannot change CONTIGUOUS attribute on use-associated 'w'
  contiguous w
  !ERROR: 'z' is use-associated from module 'm2' and cannot be re-declared
  integer z
  real, target :: a(10)
  real, contiguous, pointer :: p(:) => a
  !ERROR: Reference to 'y' is ambiguous
  y = 1
 contains
  subroutine inner
    p(1) = 0. ! ok - check for regression on contiguous host assoc.
  end subroutine
end
