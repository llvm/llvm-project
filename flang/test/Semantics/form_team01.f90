! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in form team statements

subroutine test
  use, intrinsic :: iso_fortran_env, only: team_type
  type(team_type) :: team
  integer :: team_number
  integer :: team_index
  integer :: statvar
  character(len=50) :: errvar
  integer, codimension[*] :: co_team_number
  integer, codimension[*] :: co_team_index
  type(team_type), dimension(1) :: array_team
  integer, dimension(1) :: array_team_number
  integer, dimension(1) :: array_team_index
  integer, dimension(1) :: array_statvar
  character(len=50), dimension(1) :: array_errvar

  logical :: invalid_argument

  ! Valid invocations which should produce no errors.
  FORM TEAM (team_number, team)
  ! One form-team-spec argument.
  FORM TEAM (team_number, team, NEW_INDEX=team_index)
  FORM TEAM (team_number, team, STAT=statvar)
  FORM TEAM (team_number, team, ERRMSG=errvar)
  ! Two form-team-spec arguments in any order.
  FORM TEAM (team_number, team, NEW_INDEX=team_index, STAT=statvar)
  FORM TEAM (team_number, team, STAT=statvar, NEW_INDEX=team_index)
  FORM TEAM (team_number, team, NEW_INDEX=team_index, ERRMSG=errvar)
  FORM TEAM (team_number, team, ERRMSG=errvar, NEW_INDEX=team_index)
  FORM TEAM (team_number, team, STAT=statvar, ERRMSG=errvar)
  FORM TEAM (team_number, team, ERRMSG=errvar, STAT=statvar)
! Three form-team-spec arguments in any order.
  FORM TEAM (team_number, team, NEW_INDEX=team_index, STAT=statvar, ERRMSG=errvar) ! identity
  FORM TEAM (team_number, team, STAT=statvar, NEW_INDEX=team_index, ERRMSG=errvar) ! transposition (1,2)
  FORM TEAM (team_number, team, ERRMSG=errvar, STAT=statvar, NEW_INDEX=team_index) ! transposition (1,3)
  FORM TEAM (team_number, team, NEW_INDEX=team_index, ERRMSG=errvar, STAT=statvar) ! transposition (2,3)
  FORM TEAM (team_number, team, ERRMSG=errvar, NEW_INDEX=team_index, STAT=statvar) ! cycle (1,2,3)
  FORM TEAM (team_number, team, STAT=statvar, ERRMSG=errvar, NEW_INDEX=team_index) ! cycle (1,3,2)
  ! It is semantically legal for team_index to be coindexed.
  FORM TEAM (team_number, team, NEW_INDEX=co_team_index[this_image()])

  ! Semantically invalid invocations.
  !ERROR: Must have INTEGER type, but is LOGICAL(4)
  FORM TEAM (invalid_argument, team)
  !ERROR: Must have INTEGER type, but is REAL(4)
  FORM TEAM (0.0, team)
  !ERROR: Must have INTEGER type, but is LOGICAL(4)
  FORM TEAM (team_number, team, NEW_INDEX=invalid_argument)
  !ERROR: Must have INTEGER type, but is LOGICAL(4)
  FORM TEAM (team_number, team, STAT=invalid_argument)
  !ERROR: Must have CHARACTER type, but is LOGICAL(4)
  FORM TEAM (team_number, team, ERRMSG=invalid_argument)

  ! Arguments with rank mismatches.
  !ERROR: Must be a scalar value, but is a rank-1 array
  FORM TEAM (array_team_number, team)
  !ERROR: Must be a scalar value, but is a rank-1 array
  FORM TEAM (team_number, array_team)
  !ERROR: Must be a scalar value, but is a rank-1 array
  FORM TEAM (team_number, team, NEW_INDEX=array_team_index)
  !ERROR: Must be a scalar value, but is a rank-1 array
  FORM TEAM (team_number, team, STAT=array_statvar)
  !ERROR: Must be a scalar value, but is a rank-1 array
  FORM TEAM (team_number, team, ERRMSG=array_errvar)
end subroutine
