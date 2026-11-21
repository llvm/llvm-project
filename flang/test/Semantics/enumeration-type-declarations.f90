! RUN: %python %S/test_errors.py %s %flang_fc1
! Test declaration, constructor, and expression semantics for enumeration types

! C7114: access specifier only allowed in module
subroutine test_access_specifier_outside_module()
  !ERROR: PRIVATE attribute may only appear in the specification part of a module
  !ERROR: Access specifier on ENUMERATION TYPE may only appear in the specification part of a module
  enumeration type, private :: color
    enumerator :: red, green, blue
  end enumeration type
end subroutine

! Valid: basic declarations and usage
subroutine test_basic_declarations()
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type

  type(color) :: c1, c2
  logical :: l

  ! Valid: assign an enumerator
  c1 = red
  c2 = blue

  ! Valid: comparison produces logical
  l = (c1 == c2)
  l = (c1 /= red)
end subroutine

! Valid: constructor syntax — color(n) where n is a positive integer <= count
subroutine test_constructor_valid()
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type

  type(color) :: c

  ! Valid: integer constructor in range
  c = color(1)
  c = color(2)
  c = color(3)
end subroutine

! Constructor errors
subroutine test_constructor_errors()
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type

  type(color) :: c

  ! ERROR: Enumeration constructor for 'color' requires exactly one argument
  c = color()

  ! ERROR: Enumeration constructor for 'color' requires exactly one argument
  c = color(1, 2)

  ! ERROR: Enumeration constructor for 'color' may not have a keyword argument
  c = color(val=1)

  ! ERROR: Enumeration constructor argument must be INTEGER, but is REAL(4)
  c = color(1.0)

  ! ERROR: Enumeration constructor argument for 'color' must be scalar
  c = color([1, 2])

  ! ERROR: Enumeration constructor value (0) for 'color' must be positive and less than or equal to the number of enumerators (3)
  c = color(0)

  ! ERROR: Enumeration constructor value (4) for 'color' must be positive and less than or equal to the number of enumerators (3)
  c = color(4)
end subroutine

! Component reference on enumeration type is not allowed
subroutine test_component_reference()
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type

  type(color) :: c
  integer :: i

  c = red
  ! ERROR: Component reference is not allowed for enumeration type 'color'
  i = c%__ordinal
end subroutine

! Module providing an enumeration type by USE association
module enum_constructor_mod
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
end module

! Constructor errors for a USE-associated enumeration type.
! This exercises the cross-module path: the type's local symbol carries
! UseDetails, so the enumeration-specific checks must follow USE association.
subroutine test_constructor_errors_use()
  use enum_constructor_mod

  type(color) :: c

  ! ERROR: Enumeration constructor for 'color' requires exactly one argument
  c = color()

  ! ERROR: Enumeration constructor for 'color' requires exactly one argument
  c = color(1, 2)

  ! ERROR: Enumeration constructor for 'color' may not have a keyword argument
  c = color(val=1)

  ! ERROR: Enumeration constructor argument must be INTEGER, but is REAL(4)
  c = color(1.0)

  ! ERROR: Enumeration constructor argument for 'color' must be scalar
  c = color([1, 2])

  ! ERROR: Enumeration constructor value (0) for 'color' must be positive and less than or equal to the number of enumerators (3)
  c = color(0)

  ! ERROR: Enumeration constructor value (4) for 'color' must be positive and less than or equal to the number of enumerators (3)
  c = color(4)
end subroutine

! A duplicate enumerator name within an enumeration type is diagnosed as a
! clean semantic error.
subroutine test_duplicate_enumerator()
  enumeration type :: color
    !ERROR: 'red' is already declared in this scoping unit
    enumerator :: red, green, red
  end enumeration type
end subroutine

! An enumeration type has no implicit conversion to or from INTEGER, and is not
! numeric: assignment and arithmetic mixing an enumeration value with an
! integer (or two enumeration values) must be rejected.
subroutine test_assignment_and_arithmetic()
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
  type(color) :: c
  integer :: i

  ! No implicit enumeration -> integer conversion (use INT()).
  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types INTEGER(4) and TYPE(color)
  i = red

  ! No implicit integer -> enumeration conversion (use the constructor).
  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types TYPE(color) and INTEGER(4)
  c = 1

  ! Enumeration values are not numeric.
  !ERROR: Operands of + must be numeric; have TYPE(color) and INTEGER(4)
  i = red + 1
  !ERROR: Operands of + must be numeric; have TYPE(color) and TYPE(color)
  c = red + green
end subroutine

! An enumeration type has no type parameters (F2023 7.6.2), so no type-spec may
! give it a type-param-spec-list.  An enumeration constructor is still valid.
subroutine test_type_parameters(u)
  enumeration type :: state
    enumerator :: off, on
  end enumeration type
  type :: holder
    !ERROR: Enumeration type 'state' has no type parameters
    type(state(1)) :: comp
  end type
  class(*), intent(in) :: u
  !ERROR: Enumeration type 'state' has no type parameters
  type(state(1)) :: a
  !ERROR: Enumeration type 'state' has no type parameters
  type(state(k=2)) :: b
  !ERROR: Enumeration type 'state' has no type parameters
  type(state(:)), allocatable :: c
  class(*), allocatable :: d

  a = state(1)
  b = off
  !ERROR: Enumeration type 'state' has no type parameters
  allocate(state(1) :: d)
  deallocate(d)
  allocate(d, source=state(2))
  select type (u)
  !ERROR: Enumeration type 'state' has no type parameters
  type is (state(1))
  end select
end subroutine

! A FUNCTION prefix and IMPLICIT resolve their type-spec on separate paths.
module enum_type_param_mod
  enumeration type :: state
    enumerator :: off, on
  end enumeration type
contains
  !ERROR: Enumeration type 'state' has no type parameters
  type(state(1)) function param_result()
  end function
end module

subroutine test_type_parameters_implicit()
  use enum_type_param_mod, only: state
  !ERROR: Enumeration type 'state' has no type parameters
  implicit type(state(1)) (z)
end subroutine
