! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in sync team statements.
! Some of the errors in this test would be hidden by the errors in
! the test synchronization04a.f90 if they were included in that file,
! and are thus tested here.

program test_sync_team
  use iso_fortran_env, only : team_type
  implicit none

  integer sync_status, co_indexed_integer[*], superfluous_stat, non_scalar(1), not_a_team
  character(len=128) error_message, co_indexed_character[*], superfluous_errmsg
  logical invalid_type
  type(team_type) warriors

  !___ non-standard-conforming statements ___

  !ERROR: Team value must be of type TEAM_TYPE from module ISO_FORTRAN_ENV
  sync team(not_a_team)

  !ERROR: Must have INTEGER type, but is LOGICAL(4)
  sync team(warriors, stat=invalid_type)

  !ERROR: Must be a scalar value, but is a rank-1 array
  sync team(warriors, stat=non_scalar)

  !ERROR: Must have CHARACTER type, but is LOGICAL(4)
  sync team(warriors, errmsg=invalid_type)

  ! No specifier shall appear more than once in a given sync-stat-list
  sync team(warriors, stat=sync_status, stat=superfluous_stat)

  ! No specifier shall appear more than once in a given sync-stat-list
  sync team(warriors, errmsg=error_message, errmsg=superfluous_errmsg)

  ! Fortran 2018 standard C1173: `stat` shall not be coindexed
  sync team(warriors, stat=co_indexed_integer[1])

  ! Fortran 2018 standard C1173: `errmsg` shall not be coindexed
  sync team(warriors, errmsg=co_indexed_character[1])

end program test_sync_team
