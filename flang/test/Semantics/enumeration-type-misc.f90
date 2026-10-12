! RUN: %python %S/test_errors.py %s %flang_fc1 -fenumeration-type
! Miscellaneous enumeration-type use cases: a single-enumerator type, MERGE over
! enumeration values, and enumeration-type argument association.

module enum_misc_mod
  !WARNING: ENUMERATION TYPE support is incomplete and should be enabled only for testing
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type

  !WARNING: ENUMERATION TYPE support is incomplete and should be enabled only for testing
  enumeration type :: direction
    enumerator :: north, south
  end enumeration type

  !WARNING: ENUMERATION TYPE support is incomplete and should be enabled only for testing
  enumeration type :: single
    enumerator :: only
  end enumeration type

contains
  ! Valid: a single-enumerator type.
  subroutine test_single()
    type(single) :: x
    x = only
  end subroutine

  ! Valid: MERGE selects between two enumerators of the same type.
  subroutine test_merge()
    type(color) :: c
    c = merge(red, green, .true.)
  end subroutine

  subroutine take_color(c)
    type(color), intent(in) :: c
  end subroutine

  ! Valid: passing a matching enumeration type.
  subroutine test_arg_ok()
    type(color) :: c
    c = red
    call take_color(c)
  end subroutine

  ! A different enumeration type is not compatible with the dummy argument.
  subroutine test_arg_mismatch()
    type(direction) :: d
    d = north
    !ERROR: Actual argument type 'direction' is not compatible with dummy argument type 'color'
    call take_color(d)
  end subroutine
end module
