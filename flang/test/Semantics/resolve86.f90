! RUN: %python %S/test_errors.py %s %flang_fc1
! C736 If EXTENDS appears and the type being defined has a coarray ultimate 
! component, its parent type shall have a coarray ultimate component.
!
subroutine s()
  type coarrayParent
    real,allocatable, codimension[:] :: parentField
  end type coarrayParent

  type, extends(coarrayParent) :: goodChildType
    real, allocatable, codimension[:] :: childField
  end type goodChildType

  type, extends(coarrayParent) :: brotherType
    real :: brotherField
  end type brotherType

  type, extends(brotherType) :: grandChildType
    real, allocatable, codimension[:] :: grandChildField
  end type grandChildType

  type plainParent
  end type plainParent

  !ERROR: Type 'badchildtype' has a coarray ultimate component so the type at the base of its type extension chain ('plainparent') must be a type that has a coarray ultimate component
  type, extends(plainParent) :: badChildType
    real, allocatable, codimension[:] :: childField
  end type badChildType

  type, extends(plainParent) :: plainChild
    real :: realField
  end type plainChild

  !ERROR: Type 'badchildtype2' has a coarray ultimate component so the type at the base of its type extension chain ('plainparent') must be a type that has a coarray ultimate component
  type, extends(plainChild) :: badChildType2
    real, allocatable, codimension[:] :: childField
  end type badChildType2

  !ERROR: Type 'badchildtype3' has a coarray ultimate component so the type at the base of its type extension chain ('plainparent') must be a type that has a coarray ultimate component
  type, extends(plainParent) :: badChildType3
    type(coarrayParent) :: childField
  end type badChildType3

end subroutine s
