! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
module m
  abstract interface
    subroutine foo
    end subroutine
    subroutine foo2
    end subroutine
  end interface

  procedure() :: a
  procedure(integer) :: b
  procedure(foo) :: c
  procedure(bar) :: d
  !ERROR: 'missing' must be an abstract interface or a procedure with an explicit interface
  procedure(missing) :: e
  !ERROR: 'b' must be an abstract interface or a procedure with an explicit interface
  procedure(b) :: f
  procedure(c) :: g
  external :: h
  !ERROR: 'h' must be an abstract interface or a procedure with an explicit interface
  procedure(h) :: i
  procedure(forward) :: j
  !ERROR: 'bad1' must be an abstract interface or a procedure with an explicit interface
  !ERROR: Procedure 'k1' may not be an array without an explicit interface
  procedure(bad1) :: k1
  !ERROR: 'bad2' must be an abstract interface or a procedure with an explicit interface
  procedure(bad2) :: k2
  !ERROR: 'bad3' must be an abstract interface or a procedure with an explicit interface
  procedure(bad3) :: k3

  abstract interface
    subroutine forward
    end subroutine
  end interface

  real :: bad1(1)
  real :: bad2
  type :: bad3
  end type

  !PORTABILITY: Name 'm' declared in a module should not have the same name as the module
  type :: m
  end type m

  !ERROR: EXTERNAL attribute was already specified on 'a'
  !ERROR: EXTERNAL attribute was already specified on 'b'
  !ERROR: EXTERNAL attribute was already specified on 'c'
  !ERROR: EXTERNAL attribute was already specified on 'd'
  external :: a, b, c, d
  !ERROR: EXTERNAL attribute not allowed on 'm'
  external :: m
  !WARNING: EXTERNAL attribute was already specified on 'foo'
  external :: foo
  !ERROR: EXTERNAL attribute not allowed on 'bar'
  external :: bar

  !ERROR: An entity may not have the ASYNCHRONOUS attribute unless it is a variable
  asynchronous :: async
  external :: async

  !ERROR: PARAMETER attribute not allowed on 'm'
  parameter(m=2)
  !ERROR: PARAMETER attribute not allowed on 'foo'
  parameter(foo=2)
  !ERROR: PARAMETER attribute not allowed on 'bar'
  parameter(bar=2)

  type, abstract :: t1
    integer :: i
  contains
    !ERROR: 'proc' must be an abstract interface or a procedure with an explicit interface
    !ERROR: Procedure component 'p1' must have NOPASS attribute or explicit interface
    procedure(proc), deferred :: p1
  end type t1

  abstract interface
    function f()
    end function
  end interface

contains
  subroutine bar
  end subroutine
  !ERROR: An entity may not have the ASYNCHRONOUS attribute unless it is a variable
  subroutine test
    asynchronous test
    !ERROR: Abstract procedure interface 'foo2' may not be referenced
    call foo2()
    !ERROR: Abstract procedure interface 'f' may not be referenced
    x = f()
  end subroutine
end module
