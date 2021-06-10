! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
module m
! C743 No component-attr-spec shall appear more than once in a 
! given component-def-stmt.
!
! R737 data-component-def-stmt ->
!        declaration-type-spec [[, component-attr-spec-list] ::]
!        component-decl-list
!  component-attr-spec values are:
!    PUBLIC, PRIVATE, ALLOCATABLE, CODIMENSION [*], CONTIGUOUS, DIMENSION(5), 
!      POINTER

  type :: derived
    !WARNING: Attribute 'PUBLIC' cannot be used more than once
    real, public, allocatable, public :: field1
    !WARNING: Attribute 'PRIVATE' cannot be used more than once
    real, private, allocatable, private :: field2
    !ERROR: Attributes 'PUBLIC' and 'PRIVATE' conflict with each other
    real, public, allocatable, private :: field3
    !WARNING: Attribute 'ALLOCATABLE' cannot be used more than once
    real, allocatable, public, allocatable :: field4
    !ERROR: Attribute 'CODIMENSION' cannot be used more than once
    real, public, codimension[:], allocatable, codimension[:] :: field5
    !WARNING: Attribute 'CONTIGUOUS' cannot be used more than once
    real, public, contiguous, pointer, contiguous, dimension(:) :: field6
    !ERROR: Attribute 'DIMENSION' cannot be used more than once
    real, dimension(5), public, dimension(5) :: field7
    !WARNING: Attribute 'POINTER' cannot be used more than once
    real, pointer, public, pointer :: field8
  end type derived

end module m
