! RUN: %python %S/test_errors.py %s %flang_fc1 -fenumeration-type
! Forward reference to an enumeration type.
!
! F2023 C7116: the enumeration-type-name in an enumeration-type-spec shall be
! the name of a previously defined enumeration type.  Unlike the derived type
! on which enumeration types are based, an enumeration type may not be forward
! referenced.

program p
  !ERROR: Enumeration type 'color' must be defined before it is referenced
  implicit type(color) (c)        ! forward reference: 'color' defined below

  !WARNING: ENUMERATION TYPE support is incomplete and should be enabled only for testing
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type

  c1 = red
end program
