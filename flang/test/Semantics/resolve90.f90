! RUN: %S/test_errors.sh %s %t %f18
! C751 A component shall not have both the ALLOCATABLE and POINTER attributes.
! C752 If the CONTIGUOUS attribute is specified, the component shall be an 
!   array with the POINTER attribute.
! C753 The * char-length option is permitted only if the component is of type 
!   character.
subroutine s()
  type derivedType
    !ERROR: 'pointerallocatablefield' may not have both the POINTER and ALLOCATABLE attributes
    real, pointer, allocatable :: pointerAllocatableField
    real, dimension(:), contiguous, pointer :: goodContigField
    !ERROR: A CONTIGUOUS component must be an array with the POINTER attribute
    real, dimension(:), contiguous, allocatable :: badContigField
    character :: charField * 3
    !ERROR: A length specifier cannot be used to declare the non-character entity 'realfield'
    real :: realField * 3
  end type derivedType
end subroutine s
