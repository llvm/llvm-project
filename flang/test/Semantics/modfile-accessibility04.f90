! RUN: %python %S/test_errors.py %s %flang_fc1
! Test that explicit PUBLIC/PRIVATE on individual symbols overrides module-level accessibility
module basemod
  implicit none
  integer :: var1 = 1
  integer :: var2 = 2
  integer :: var3 = 3
end module

module middlemod
  use basemod
  implicit none
  private basemod  ! Make all entities from basemod private by default
  public :: var2   ! But explicitly make var2 public
end module

program main
  ! var2 should be accessible because of explicit "public :: var2"
  use middlemod, only: var2
  ! ERROR: 'var1' is PRIVATE in 'middlemod'
  use middlemod, only: var1
  ! ERROR: 'var3' is PRIVATE in 'middlemod'
  use middlemod, only: var3
  implicit none
end
