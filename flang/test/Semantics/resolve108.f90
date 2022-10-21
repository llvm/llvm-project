! RUN: %python %S/test_errors.py %s %flang_fc1
! Tests attempts at forward references to local names in a FUNCTION prefix

! This case is not an error, but will elicit bogus errors if the
! result type of the function is badly resolved.
module m1
  type t1
    sequence
    integer not_m
  end type
 contains
  type(t1) function foo(n)
    integer, intent(in) :: n
    type t1
      sequence
      integer m
    end type
    foo%m = n
  end function
end module

subroutine s1
  use :: m1, only: foo
  type t1
    sequence
    integer m
  end type
  type(t1) x
  x = foo(234)
  print *, x
end subroutine

module m2
  integer, parameter :: k = kind(1.e0)
 contains
  real(kind=k) function foo(n)
    integer, parameter :: k = kind(1.d0)
    integer, intent(in) :: n
    foo = n
  end function
end module

subroutine s2
  use :: m2, only: foo
  !If we got the type of foo right, this declaration will fail
  !due to an attempted division by zero.
  !WARNING: INTEGER(4) division by zero
  !ERROR: Must be a constant value
  integer, parameter :: test = 1 / (kind(foo(1)) - kind(1.d0))
end subroutine

module m3
  real(kind=kind(1.0e0)) :: x
 contains
  real(kind=kind(x)) function foo(x)
    real(kind=kind(1.0d0)) x
    !WARNING: INTEGER(4) division by zero
    !ERROR: Must be a constant value
    integer, parameter :: test = 1 / (kind(foo) - kind(1.d0))
    foo = n
  end function
end module

module m4
 contains
  real(n) function foo(x)
    !ERROR: 'foo' is not an object that can appear in an expression
    integer, parameter :: n = kind(foo)
    real(n), intent(in) :: x
    !ERROR: 'x' is not an object that can appear in an expression
    foo = x
  end function
end module
