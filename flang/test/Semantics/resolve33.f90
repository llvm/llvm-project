! Derived type parameters

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
end module
