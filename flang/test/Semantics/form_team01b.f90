! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
! Check for semantic errors in form team statements
! This subtest contains tests for unimplemented errors.

subroutine test
  use, intrinsic :: iso_fortran_env, only: team_type
  type(team_type) :: team
  integer :: team_number
  integer, codimension[*] :: co_statvar
  character(len=50), codimension[*] :: co_errvar

  ! Semantically invalid invocations.
  ! argument 'stat' shall not be a coindexed object
  !ERROR: to be determined
  FORM TEAM (team_number, team, STAT=co_statvar[this_image()])
  ! argument 'errmsg' shall not be a coindexed object
  !ERROR: to be determined
  FORM TEAM (team_number, team, ERRMSG=co_errvar[this_image()])

end subroutine
