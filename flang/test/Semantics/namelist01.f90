! RUN: %python %S/test_errors.py %s %flang_fc1
! Test for checking namelist constraints, C8103-C8105

module dup
  integer dupName
  integer uniqueName
end module dup

subroutine C8103a(x)
  use dup, only: uniqueName, dupName
  integer :: x
  !ERROR: 'dupname' is already declared in this scoping unit
  namelist /dupName/ x, x
end subroutine C8103a

subroutine C8103b(y)
  use dup, only: uniqueName
  integer :: y
  namelist /dupName/ y, y
end subroutine C8103b

subroutine C8104a(ivar, jvar)
  integer :: ivar(10,8)
  integer :: jvar(*)
  NAMELIST /NLIST/ ivar
  !ERROR: A namelist group object 'jvar' must not be assumed-size
  NAMELIST /NLIST/ jvar
end subroutine C8104a

subroutine C8104b(ivar, jvar)
  integer, dimension(*) :: jvar
  !ERROR: A namelist group object 'jvar' must not be assumed-size
  NAMELIST /NLIST/ ivar, jvar
end subroutine C8104b

subroutine C8104c(jvar)
  integer :: jvar(10, 3:*)
  !ERROR: A namelist group object 'jvar' must not be assumed-size
  NAMELIST /NLIST/ jvar
end subroutine C8104c

module C8105
  integer, private :: x
  public :: NLIST
  !ERROR: A PRIVATE namelist group object 'x' must not be in a PUBLIC namelist
  NAMELIST /NLIST/ x
  !ERROR: A PRIVATE namelist group object 'x' must not be in a PUBLIC namelist
  NAMELIST /NLIST2/ x
  public :: NLIST2
end module C8105
