! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
module m
 contains
  subroutine foo(a)
    real, intent(in), target :: a(:)
  end subroutine
end module

program test
  use m
  real, target :: a(1)
  real :: b(1)
  call foo(a) ! ok
  !WARNING: Any pointer associated with TARGET dummy argument 'a=' during this call must not be used afterwards, as 'b' is not a target
  call foo(b)
  !WARNING: Any pointer associated with TARGET dummy argument 'a=' during this call will not be associated with the value of '(a)' afterwards
  call foo((a))
  !WARNING: Any pointer associated with TARGET dummy argument 'a=' during this call will not be associated with the value of 'a([INTEGER(8)::1_8])' afterwards
  call foo(a([1]))
  !ERROR: Scalar actual argument may not be associated with assumed-shape dummy argument 'a='
  call foo(a(1))
end
