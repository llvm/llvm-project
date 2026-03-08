! RUN: %flang_fc1 -fsyntax-only %s
! Test that entities from a module with PUBLIC modulename remain accessible
module basemod
  implicit none
  private  !-- Default is private
  integer, public :: base_public_var = 42
  integer         :: base_private_var = 99
end module

module middlemod
  use basemod
  implicit none
  private         !-- Default is private
  public basemod  !-- But entities from basemod should remain public
  integer :: middle_private_var = 100
end module

program main
  use middlemod
  implicit none
  integer :: x
  ! base_public_var should be accessible because of "public basemod"
  x = base_public_var
  ! This should compile without errors
  print *, x
end program
