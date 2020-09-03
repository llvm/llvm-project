! RUN: %S/test_errors.sh %s %t %f18
! Derived type parameters
! C731 The same type-param-name shall not appear more than once in a given
! derived-type-stmt.
! C741 A type-param-name in a type-param-def-stmt in a derived-type-def shall
! be one of the type-paramnames in the derived-type-stmt of that
! derived-type-def.
! C742 Each type-param-name in the derived-type-stmt in a derived-type-def
! shall appear exactly once as a type-param-name in a type-param-def-stmt 
! in that derived-type-def.

module m
  !ERROR: Duplicate type parameter name: 'a'
  type t1(a, b, a)
    integer, kind :: a
    integer(8), len :: b
  end type
  !ERROR: No definition found for type parameter 'b'
  type t2(a, b, c)
    integer, kind :: a
    integer, len :: c
  end type
  !ERROR: No definition found for type parameter 'b'
  type t3(a, b)
    integer, kind :: a
    integer :: b
  end type
  type t4(a)
    integer, kind :: a
    !ERROR: 'd' is not a type parameter of this derived type
    integer(8), len :: d
  end type
  type t5(a, b)
    integer, len :: a
    integer, len :: b
    !ERROR: Type parameter, component, or procedure binding 'a' already defined in this type
    integer, len :: a
  end type
  !ERROR: No definition found for type parameter 'k'
  !ERROR: No definition found for type parameter 'l'
  type :: t6(k, l)
    character(kind=k, len=l) :: d3
  end type
  type(t6(2, 10)) :: x3
end module
