! RUN: %python %S/test_errors.py %s %flang_fc1
! Tests valid and invalid NULL initializers

module m1
  implicit none
  !ERROR: No explicit type declared for 'null'
  private :: null
end module

module m2
  implicit none
  private :: null
  integer, pointer :: p => null()
end module

module m3
  private :: null
  integer, pointer :: p => null()
end module

module m4
  intrinsic :: null
  integer, pointer :: p => null()
end module

module m5
  external :: null
  !ERROR: Pointer initializer must be intrinsic NULL()
  integer, pointer :: p => null()
end module

module m6
  !ERROR: Symbol 'null' cannot have both INTRINSIC and EXTERNAL attributes
  integer, pointer :: p => null()
  external :: null
end module

module m7
  interface
    function null() result(p)
      integer, pointer :: p
    end function
  end interface
  !ERROR: Pointer initializer must be intrinsic NULL()
  integer, pointer :: p => null()
end module

module m8
  integer, pointer :: p => null()
  interface
    !ERROR: 'null' is already declared in this scoping unit
    function null() result(p)
      integer, pointer :: p
    end function
  end interface
end module

module m9a
  intrinsic :: null
 contains
  function foo()
    integer, pointer :: foo
    foo => null()
  end function
end module
module m9b
  use m9a, renamed => null, null => foo
  integer, pointer :: p => renamed()
  !ERROR: Pointer initializer must be intrinsic NULL()
  integer, pointer :: q => null()
  integer, pointer :: d1, d2
  data d1/renamed()/
  !ERROR: An initial data target must be a designator with constant subscripts
  data d2/null()/
end module

subroutine m10
  real, pointer :: x, y
  !ERROR: 'null' must be an array or structure constructor if used with non-empty parentheses as a DATA statement constant
  data x/null(y)/
end

subroutine m11
  type :: null
    integer :: mold
  end type
  type(null) :: obj(2)
  integer, parameter :: j = 0
  data obj/null(mold=j), null(j)/ ! both fine
end subroutine

subroutine m12
  integer, parameter :: j = 1
  integer, target, save :: null(1)
  integer, pointer :: p
  data p/null(j)/ ! ok
end subroutine
