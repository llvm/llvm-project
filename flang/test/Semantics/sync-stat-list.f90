! RUN: %python %S/test_errors.py %s %flang_fc1
! There are sync-stat-lists in critical-stmt, sync-all-stmt,
! sync-images-stmt, sync-memory-stmt, sync-team-stmt,
! event-post-stmt, unlock-stmt, change-team-stmt, and end-change-team-stmt.
!
! Some of these statements have their sync-stat-lists tested in other tests.
! This test contains the statements that do not, namely critical-stmt, unlock-stmt,
! change-team-stmt, and end-change-team-stmt.

program test_sync_stat_list
  use iso_fortran_env, only: team_type, lock_type

  implicit none

  integer coarray[*], sync_status, non_scalar(2), superfluous_stat, coindexed_integer[*]
  character(len=128) error_message, superfluous_errmsg, coindexed_character[*]
  logical invalid_type
  type(team_type) :: home
  type(lock_type) :: latch

  ! valid
  change team (home, stat=sync_status, errmsg=error_message)
  end team (stat=sync_status, errmsg=error_message)

  !ERROR: Must be a scalar value, but is a rank-1 array
  change team (home, stat=non_scalar, errmsg=error_message)
  end team

  !ERROR: Must have INTEGER type, but is LOGICAL(4)
  change team (home, stat=invalid_type, errmsg=error_message)
  end team

  !ERROR: Must have CHARACTER type, but is LOGICAL(4)
  change team (home, stat=sync_status, errmsg=invalid_type)
  end team

  change team (home)
  !ERROR: Must be a scalar value, but is a rank-1 array
  end team (stat=non_scalar, errmsg=error_message)

  change team (home)
  !ERROR: Must have INTEGER type, but is LOGICAL(4)
  end team (stat=invalid_type, errmsg=error_message)

  change team (home)
  !ERROR: Must have CHARACTER type, but is LOGICAL(4)
  end team (stat=sync_status, errmsg=invalid_type)

  !ERROR: The stat-variable in a sync-stat-list may not be repeated
  change team (home, stat=sync_status, errmsg=error_message, stat=superfluous_stat)
  end team

  !ERROR: The errmsg-variable in a sync-stat-list may not be repeated
  change team (home, stat=sync_status, errmsg=error_message, errmsg=superfluous_errmsg)
  end team

  change team (home)
  !ERROR: The stat-variable in a sync-stat-list may not be repeated
  end team (stat=sync_status, errmsg=error_message, stat=superfluous_stat)

  change team (home)
  !ERROR: The errmsg-variable in a sync-stat-list may not be repeated
  end team (stat=sync_status, errmsg=error_message, errmsg=superfluous_errmsg)

  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  change team (home, stat=coindexed_integer[1], errmsg=coindexed_character[1])
  end team

  change team (home)
  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  end team (stat=coindexed_integer[1], errmsg=coindexed_character[1])

  ! valid
  unlock (latch, stat=sync_status, errmsg=error_message)

  !ERROR: Must be a scalar value, but is a rank-1 array
  unlock (latch, stat=non_scalar, errmsg=error_message)

  !ERROR: Must have INTEGER type, but is LOGICAL(4)
  unlock (latch, stat=invalid_type, errmsg=error_message)

  !ERROR: Must have CHARACTER type, but is LOGICAL(4)
  unlock (latch, stat=sync_status, errmsg=invalid_type)

  !ERROR: The stat-variable in a sync-stat-list may not be repeated
  unlock (latch, stat=sync_status, stat=superfluous_stat)

  !ERROR: The errmsg-variable in a sync-stat-list may not be repeated
  unlock (latch, stat=sync_status, errmsg=error_message, errmsg=superfluous_errmsg)

  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  unlock (latch, stat=coindexed_integer[1], errmsg=coindexed_character[1])

  ! valid
  critical (stat=sync_status, errmsg=error_message)
  end critical

  !ERROR: Must be a scalar value, but is a rank-1 array
  critical (stat=non_scalar, errmsg=error_message)
  end critical

  !ERROR: Must have INTEGER type, but is LOGICAL(4)
  critical (stat=invalid_type, errmsg=error_message)
  end critical

  !ERROR: Must have CHARACTER type, but is LOGICAL(4)
  critical (stat=sync_status, errmsg=invalid_type)
  end critical

  !ERROR: The stat-variable in a sync-stat-list may not be repeated
  critical (stat=sync_status, errmsg=error_message, stat=superfluous_stat)
  end critical

  !ERROR: The errmsg-variable in a sync-stat-list may not be repeated
  critical (stat=sync_status, errmsg=error_message, errmsg=superfluous_errmsg)
  end critical

  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  critical (stat=coindexed_integer[1], errmsg=coindexed_character[1])
  end critical

end program test_sync_stat_list
