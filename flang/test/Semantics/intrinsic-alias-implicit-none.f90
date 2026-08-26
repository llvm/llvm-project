! RUN: %flang_fc1 -fsyntax-only %s
!
! Check that GNU Fortran compatibility intrinsics IARGC and GETARG
! are recognized as intrinsic procedures under IMPLICIT NONE.

program test_intrinsic_aliases
  implicit none

  integer :: n
  character(len=100) :: arg

  ! IARGC is an alias for COMMAND_ARGUMENT_COUNT
  n = iargc()

  ! GETARG is an alias for GET_COMMAND_ARGUMENT
  call getarg(1, arg)
end program
