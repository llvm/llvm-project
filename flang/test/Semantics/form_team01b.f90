! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in form team statements
! This subtest contains tests for unimplemented errors.

subroutine test
  use, intrinsic :: iso_fortran_env, only: team_type
  type(team_type) :: team
  integer :: team_number
  integer, save, codimension[*] :: co_statvar
  character(len=50), save, codimension[*] :: co_errvar
  procedure(type(team_type)) teamfunc
  !ERROR: The stat-variable or errmsg-variable in a form-team-spec-list may not be a coindexed object
  FORM TEAM (team_number, team, STAT=co_statvar[this_image()])
  !ERROR: The stat-variable or errmsg-variable in a form-team-spec-list may not be a coindexed object
  FORM TEAM (team_number, team, ERRMSG=co_errvar[this_image()])
  !ERROR: Team must be a variable in this context
  form team (team_number, teamfunc())
end subroutine
