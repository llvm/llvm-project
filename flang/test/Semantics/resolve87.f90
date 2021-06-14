! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! C737 If EXTENDS appears and the type being defined has a potential 
! subobject component of type EVENT_TYPE or LOCK_TYPE from the intrinsic 
! module ISO_FORTRAN_ENV, its parent type shall be EVENT_TYPE or LOCK_TYPE 
! or have a potential subobject component of type EVENT_TYPE or LOCK_TYPE.
module not_iso_fortran_env
  type event_type
  end type

  type lock_type
  end type
end module

subroutine C737_a()
  use iso_fortran_env

  type lockGrandParentType
    type(lock_type) :: grandParentField
  end type lockGrandParentType

  type, extends(lockGrandParentType) :: lockParentType
    real :: parentField
  end type lockParentType

  type eventParentType
    type(event_type) :: parentField
  end type eventParentType

  type noLockParentType
  end type noLockParentType

  type, extends(lockParentType) :: goodChildType1
    type(lock_type) :: childField
  end type goodChildType1

  type, extends(lockParentType) :: goodChildType2
    type(event_type) :: childField
  end type goodChildType2

  type, extends(lock_type) :: goodChildType3
    type(event_type) :: childField
  end type goodChildType3

  type, extends(event_type) :: goodChildType4
    type(lock_type) :: childField
  end type goodChildType4

  !ERROR: Type 'badchildtype1' has an EVENT_TYPE or LOCK_TYPE component, so the type at the base of its type extension chain ('nolockparenttype') must either have an EVENT_TYPE or LOCK_TYPE component, or be EVENT_TYPE or LOCK_TYPE
  type, extends(noLockParentType) :: badChildType1
    type(lock_type) :: childField
  end type badChildType1

  !ERROR: Type 'badchildtype2' has an EVENT_TYPE or LOCK_TYPE component, so the type at the base of its type extension chain ('nolockparenttype') must either have an EVENT_TYPE or LOCK_TYPE component, or be EVENT_TYPE or LOCK_TYPE
  type, extends(noLockParentType) :: badChildType2
    type(event_type) :: childField
  end type badChildType2

  !ERROR: Type 'badchildtype3' has an EVENT_TYPE or LOCK_TYPE component, so the type at the base of its type extension chain ('nolockparenttype') must either have an EVENT_TYPE or LOCK_TYPE component, or be EVENT_TYPE or LOCK_TYPE
  type, extends(noLockParentType) :: badChildType3
    type(lockParentType) :: childField
  end type badChildType3

  !ERROR: Type 'badchildtype4' has an EVENT_TYPE or LOCK_TYPE component, so the type at the base of its type extension chain ('nolockparenttype') must either have an EVENT_TYPE or LOCK_TYPE component, or be EVENT_TYPE or LOCK_TYPE
  type, extends(noLockParentType) :: badChildType4
    type(eventParentType) :: childField
  end type badChildType4

end subroutine C737_a

subroutine C737_b()
  use not_iso_fortran_env

  type lockParentType
    type(lock_type) :: parentField
  end type lockParentType

  type noLockParentType
  end type noLockParentType

  ! actually OK since this is not the predefined lock_type
  type, extends(noLockParentType) :: notBadChildType1
    type(lock_type) :: childField
  end type notBadChildType1

  ! actually OK since this is not the predefined event_type
  type, extends(noLockParentType) :: notBadChildType2
    type(event_type) :: childField
  end type notBadChildType2

end subroutine C737_b
