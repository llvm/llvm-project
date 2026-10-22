! RUN: %flang_fc1 -fdebug-dump-symbols -fenumeration-type %s 2>&1 | FileCheck %s
! Regression test: a derived type with an enumeration-type component must be
! correctly sized once the enclosing type is instantiated (which happens as
! soon as a variable of it is declared).

subroutine test_enum_component_offset()
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
  type :: holder
    integer :: n
    type(color) :: c
  end type
  ! Declaring a variable of 'holder' instantiates it.
  type(holder) :: h
  ! CHECK: h size=8 offset={{[0-9]+}}: ObjectEntity type: TYPE(holder)
  ! CHECK: DerivedType scope: holder size=8 alignment=4
  ! CHECK: c size=4 offset=4: ObjectEntity type: TYPE(color)
  ! CHECK: n size=4 offset=0: ObjectEntity type: INTEGER(4)
end subroutine
