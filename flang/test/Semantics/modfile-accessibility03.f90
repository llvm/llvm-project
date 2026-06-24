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
  !ERROR: The name of module 'basemod' shall appear at most once in all of the ACCESS statements in a module
  public basemod
end module
