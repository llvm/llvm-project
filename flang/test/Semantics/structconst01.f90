! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Error tests for structure constructors.
! Errors caught by name resolution are tested elsewhere; these are the
! errors meant to be caught by expression semantic analysis, as well as
! acceptable use cases.
! Type parameters are used here to make the parses unambiguous.
! C796 (R756) The derived-type-spec shall not specify an abstract type (7.5.7).
!   This refers to a derived-type-spec used in a structure constructor

module module1
  type :: type1(j)
    integer, kind :: j
    integer :: n = 1
  end type type1
  type, extends(type1) :: type2(k)
    integer, kind :: k
    integer :: m
  end type type2
  type, abstract :: abstract(j)
    integer, kind :: j
    integer :: n
  end type abstract
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
  subroutine abstractarg(x)
    class(abstract(0)), intent(in) :: x
  end subroutine abstractarg
  subroutine errors
    call type1arg(type1(0)())
    call type1arg(type1(0)(1))
    call type1arg(type1(0)(n=1))
    !ERROR: Type parameter 'j' may not appear as a component of a structure constructor
    call type1arg(type1(0)(j=1))
    !ERROR: Component 'n' conflicts with another component earlier in this structure constructor
    call type1arg(type1(0)(1,n=2))
    !ERROR: Value in structure constructor lacks a component name
    call type1arg(type1(0)(n=1,2))
    !ERROR: Component 'n' conflicts with another component earlier in this structure constructor
    call type1arg(type1(0)(n=1,n=2))
    !ERROR: Unexpected value in structure constructor
    call type1arg(type1(0)(1,2))
    call type2arg(type2(0,0)(n=1,m=2))
    call type2arg(type2(0,0)(m=2))
    !ERROR: Structure constructor lacks a value for component 'm'
    call type2arg(type2(0,0)())
    call type2arg(type2(0,0)(type1=type1(0)(n=1),m=2))
    call type2arg(type2(0,0)(type1=type1(0)(),m=2))
    !ERROR: Component 'type1' conflicts with another component earlier in this structure constructor
    call type2arg(type2(0,0)(n=1,type1=type1(0)(n=2),m=3))
    !ERROR: Component 'n' conflicts with another component earlier in this structure constructor
    call type2arg(type2(0,0)(type1=type1(0)(n=1),n=2,m=3))
    !ERROR: Component 'n' conflicts with another component earlier in this structure constructor
    call type2arg(type2(0,0)(type1=type1(0)(1),n=2,m=3))
    !ERROR: Type parameter 'j' may not appear as a component of a structure constructor
    call type2arg(type2(0,0)(j=1, &
    !ERROR: Type parameter 'k' may not appear as a component of a structure constructor
      k=2,m=3))
    !ERROR: ABSTRACT derived type 'abstract' may not be used in a structure constructor
    call abstractarg(abstract(0)(n=1))
  end subroutine errors
end module module1
