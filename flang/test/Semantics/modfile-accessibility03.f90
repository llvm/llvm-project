! RUN: %python %S/test_errors.py %s %flang_fc1
! Test error when same module is given conflicting accessibility
module basemod
  implicit none
  integer :: base_var = 42
end module

module testmod
  use basemod
  implicit none
  private basemod
  !ERROR: The accessibility of entities from module 'basemod' has already been specified
  public basemod
end module
