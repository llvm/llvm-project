! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for errors in sync team statements based on the
! statement specification in section 11.6.6 of the Fortran 2018 standard.

program test_sync_team
  use iso_fortran_env, only : team_type
  implicit none

  integer sync_status
  character(len=128) error_message
  type(team_type) warriors

  !___ standard-conforming statement ___

  sync team(warriors)
  sync team(warriors, stat=sync_status)
  sync team(warriors,                   errmsg=error_message)
  sync team(warriors, stat=sync_status, errmsg=error_message)

  !___ non-standard-conforming statement ___

  !______ missing team-value _____________________

  !ERROR: expected '('
  sync team

  !ERROR: expected ')'
  sync team(stat=sync_status, errmsg=error_message)

  !______ invalid sync-stat-lists: invalid stat= ____________

  !ERROR: expected ')'
  sync team(warriors, status=sync_status)

  ! Invalid sync-stat-list: missing stat-variable
  !ERROR: expected ')'
  sync team(warriors, stat)

  ! Invalid sync-stat-list: missing 'stat='
  !ERROR: expected ')'
  sync team(warriors, sync_status)

  !______ invalid sync-stat-lists: invalid errmsg= ____________

  ! Invalid errmsg-variable keyword
  !ERROR: expected ')'
  sync team(warriors, errormsg=error_message)

  ! Invalid sync-stat-list: missing 'errmsg='
  !ERROR: expected ')'
  sync team(warriors, error_message)

  ! Invalid sync-stat-list: missing errmsg-variable
  !ERROR: expected ')'
  sync team(warriors, errmsg)

end program test_sync_team
