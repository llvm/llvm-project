! RUN: %python %S/test_errors.py %s %flang_fc1
! Test that entities from a module with PRIVATE modulename are not accessible from further USE
module basemod
  implicit none
  type :: base_type
    integer :: x
  end type
  integer :: base_var = 42
contains
  subroutine base_sub()
  end subroutine
end module

module middlemod
  use basemod
  implicit none
  private basemod  ! Make all entities from basemod private
  integer :: middle_var = 100
end module

program main
  ! ERROR: 'base_var' is PRIVATE in 'middlemod'
  use middlemod, only: base_var
  ! ERROR: 'base_sub' is PRIVATE in 'middlemod'
  use middlemod, only: base_sub
  implicit none
end
