! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in sync all statements.
! Some of the errors in this test would be hidden by the errors in
! the test synchronization01a.f90 if they were included in that file,
! and are thus tested here.

program test_sync_all
  implicit none

  integer sync_status, co_indexed_integer[*], superfluous_stat, non_scalar(1)
  character(len=128) error_message, co_indexed_character[*], superfluous_errmsg
  logical invalid_type

  !___ non-standard-conforming statements ___

  !ERROR: Must have INTEGER type, but is LOGICAL(4)
  sync all(stat=invalid_type)

  !ERROR: Must be a scalar value, but is a rank-1 array
  sync all(stat=non_scalar)

  !ERROR: Must have CHARACTER type, but is LOGICAL(4)
  sync all(errmsg=invalid_type)

  !ERROR: The stat-variable in a sync-stat-list may not be repeated
  sync all(stat=sync_status, stat=superfluous_stat)

  !ERROR: The errmsg-variable in a sync-stat-list may not be repeated
  sync all(errmsg=error_message, errmsg=superfluous_errmsg)

  !ERROR: The stat-variable in a sync-stat-list may not be repeated
  sync all(stat=sync_status, errmsg=error_message, stat=superfluous_stat)

  !ERROR: The errmsg-variable in a sync-stat-list may not be repeated
  sync all(stat=sync_status, errmsg=error_message, errmsg=superfluous_errmsg)

  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  sync all(stat=co_indexed_integer[1])

  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  sync all(errmsg=co_indexed_character[1])

end program test_sync_all
