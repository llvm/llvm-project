! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
module m
  abstract interface
    subroutine foo
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

  type :: m ! the name of a module can be used as a local identifier
  end type m

  external :: a, b, c, d
  !ERROR: EXTERNAL attribute not allowed on 'm'
  external :: m
  !ERROR: EXTERNAL attribute not allowed on 'foo'
  external :: foo
  !ERROR: EXTERNAL attribute not allowed on 'bar'
  external :: bar

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
    !ERROR: Procedure component 'p1' has invalid interface 'proc'
    procedure(proc), deferred :: p1
  end type t1

  abstract interface
    function f()
    end function
  end interface

contains
  subroutine bar
  end subroutine
  subroutine test
    !ERROR: Abstract interface 'foo' may not be called
    call foo()
    !ERROR: Abstract interface 'f' may not be called
    x = f()
  end subroutine
end module
