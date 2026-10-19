! RUN: %flang_fc1 -fsyntax-only -fenumeration-type %s
! An enumeration type used as a derived-type component: default component
! initialization, whole-structure assignment, component assignment, use of the
! component in a relational, and arrays of the containing type must all compile.

subroutine test_enum_component_usage()
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
  type :: holder
    integer :: n = 0
    type(color) :: c = red        ! default component initializer
  end type
  type(holder) :: a, b
  type(holder) :: arr(2)
  logical :: l

  b = a                            ! whole-structure assignment
  a%c = green                      ! component assignment
  l = (a%c == green)               ! component in a relational
  arr(1)%c = blue                  ! array element component
  arr(2) = a                       ! whole-element assignment
end subroutine
