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
