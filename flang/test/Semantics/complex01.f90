! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! C718 Each named constant in a complex literal constant shall be of type 
! integer or real.
subroutine s()
  integer :: ivar = 35
  integer, parameter :: iconst = 35
  real :: rvar = 68.9
  real, parameter :: rconst = 68.9
  character :: cvar = 'hello'
  character, parameter :: cconst = 'hello'
  logical :: lvar = .true.
  logical, parameter :: lconst = .true.
  complex :: cvar1 = (1, 1)
  complex :: cvar2 = (1.0, 1.0)
  complex :: cvar3 = (1.0, 1)
  complex :: cvar4 = (1, 1.0)
  complex :: cvar5 = (iconst, 1.0)
  complex :: cvar6 = (iconst, rconst)
  complex :: cvar7 = (rconst, iconst)

  !ERROR: must be a constant
  complex :: cvar8 = (ivar, 1.0)
  !ERROR: must be a constant
  !ERROR: must be a constant
  complex :: cvar9 = (ivar, rvar)
  !ERROR: must be a constant
  !ERROR: must be a constant
  complex :: cvar10 = (rvar, ivar)
  !ERROR: operands must be INTEGER or REAL
  complex :: cvar11 = (cconst, 1.0)
  !ERROR: operands must be INTEGER or REAL
  complex :: cvar12 = (lconst, 1.0)
end subroutine s
