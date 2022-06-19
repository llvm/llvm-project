! RUN: %python %S/test_errors.py %s %flang_fc1
! Test specification expressions

module m
  type :: t(n)
    integer, len :: n = 1
    character(len=n) :: c
  end type
  interface
    integer function foo()
    end function
    pure real function realfunc(x)
      real, intent(in) :: x
    end function
    pure integer function hasProcArg(p)
      import realfunc
      procedure(realfunc) :: p
      optional :: p
    end function
  end interface
  integer :: coarray[*]
 contains
  pure integer function modulefunc1(n)
    integer, value :: n
    modulefunc1 = n
  end function
  subroutine test(out, optional)
    !ERROR: Invalid specification expression: reference to impure function 'foo'
    type(t(foo())) :: x1
    integer :: local
    !ERROR: Invalid specification expression: reference to local entity 'local'
    type(t(local)) :: x2
    !ERROR: The internal function 'internal' may not be referenced in a specification expression
    type(t(internal(0))) :: x3
    integer, intent(out) :: out
    !ERROR: Invalid specification expression: reference to INTENT(OUT) dummy argument 'out'
    type(t(out)) :: x4
    integer, intent(in), optional :: optional
    !ERROR: Invalid specification expression: reference to OPTIONAL dummy argument 'optional'
    type(t(optional)) :: x5
    !ERROR: Invalid specification expression: reference to function 'hasprocarg' with dummy procedure argument 'p'
    type(t(hasProcArg())) :: x6
    !ERROR: Invalid specification expression: coindexed reference
    type(t(coarray[1])) :: x7
    type(t(kind(foo()))) :: x101 ! ok
    type(t(modulefunc1(0))) :: x102 ! ok
    type(t(modulefunc2(0))) :: x103 ! ok
   contains
    pure integer function internal(n)
      integer, value :: n
      internal = n
    end function
  end subroutine
  pure integer function modulefunc2(n)
    integer, value :: n
    modulefunc2 = n
  end function
end module
