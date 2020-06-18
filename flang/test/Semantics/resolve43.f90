! RUN: %S/test_errors.sh %s %t %f18
! Error tests for structure constructors.
! Errors caught by expression resolution are tested elsewhere; these are the
! errors meant to be caught by name resolution, as well as acceptable use
! cases.
! Type parameters are used to make the parses unambiguous.

module module1
  type :: type1(j)
    integer, kind :: j
    integer :: n = 1
  end type type1
  type, extends(type1) :: type2(k)
    integer, kind :: k
    integer :: m
  end type type2
  type :: privaten(j)
    integer, kind :: j
    integer, private :: n
  end type privaten
 contains
  subroutine type1arg(x)
    type(type1(0)), intent(in) :: x
  end subroutine type1arg
  subroutine type2arg(x)
    type(type2(0,0)), intent(in) :: x
  end subroutine type2arg
  subroutine errors
    call type1arg(type1(0)())
    call type1arg(type1(0)(1))
    call type1arg(type1(0)(n=1))
    !ERROR: Keyword 'bad=' does not name a component of derived type 'type1'
    call type1arg(type1(0)(bad=1))
    call type2arg(type2(0,0)(n=1,m=2))
    call type2arg(type2(0,0)(m=2))
    call type2arg(type2(0,0)(type1=type1(0)(n=1),m=2))
    call type2arg(type2(0,0)(type1=type1(0)(),m=2))
  end subroutine errors
end module module1

module module2
  !ERROR: No definition found for type parameter 'k'
  type :: type1(k)
  end type
  type(type1):: x
end module
