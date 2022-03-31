! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
! This test checks for semantic errors in sync team statements based on the
! statement specification in section 11.6.6 of the Fortran 2018 standard.

program test_sync_team
  use iso_fortran_env, only : team_type
  implicit none

  integer sync_status, co_indexed_integer[*], superfluous_stat, non_scalar(1), not_a_team
  character(len=128) error_message, co_indexed_character[*], superfluous_errmsg
  logical invalid_type
  type(team_type) warriors
   
  
  !___ standard-conforming statement ___

  sync team(warriors)
  sync team(warriors, stat=sync_status)
  sync team(warriors,                   errmsg=error_message)
  sync team(warriors, stat=sync_status, errmsg=error_message)
 
  !___ non-standard-conforming statement ___

  !______ missing or invalid team-value _____________________

  !ERROR: TBD
  sync team(not_a_team)

  !ERROR: expected ')'
  sync team(stat=sync_status, errmsg=error_message)

  !______ invalid sync-stat-lists: invalid stat= ____________

  !ERROR: expected ')'
  sync team(warriors, status=sync_status)

  ! Stat-variable must an integer scalar
  !ERROR: TBD
  sync team(warriors, stat=invalid_type)

  ! Stat-variable must an integer scalar
  !ERROR: TBD
  sync team(warriors, stat=non_scalar)

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

  !ERROR: TBD
  sync team(warriors, errmsg=invalid_type)

  ! Invalid sync-stat-list: missing 'errmsg='
  !ERROR: expected ')'
  sync team(warriors, error_message)

  ! Invalid sync-stat-list: missing errmsg-variable
  !ERROR: expected ')'
  sync team(warriors, errmsg)

  !______ invalid sync-stat-lists: redundant sync-stat-list ____________

  ! No specifier shall appear more than once in a given sync-stat-list
  !ERROR: to be determined
  sync team(warriors, stat=sync_status, stat=superfluous_stat)

  ! No specifier shall appear more than once in a given sync-stat-list
  !ERROR: to be determined
  sync team(warriors, errmsg=error_message, errmsg=superfluous_errmsg)
 
  !______ invalid sync-stat-lists: coindexed stat-variable ____________

  ! Check constraint C1173 from the Fortran 2018 standard
  !ERROR: to be determined
  sync team(warriors, stat=co_indexed_integer[1])
 
  ! Check constraint C1173 from the Fortran 2018 standard
  !ERROR: to be determined
  sync team(warriors, errmsg=co_indexed_character[1])

end program test_sync_team
