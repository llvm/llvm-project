! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in sync images statements.
! Some of the errors in this test would be hidden by the errors in
! the test synchronization02a.f90 if they were included in that file,
! and are thus tested here.

program test_sync_images
  implicit none

  integer, parameter :: invalid_rank(*,*) = reshape([1], [1,1])
  integer sync_status, non_scalar(2), superfluous_stat, coindexed_integer[*]
  character(len=128) error_message, superfluous_errmsg, coindexed_character[*]
  logical invalid_type

  !___ non-standard-conforming statements ___

  ! Image set shall not depend on the value of stat-variable
  sync images(sync_status, stat=sync_status)

  ! Image set shall not depend on the value of errmsg-variable
  sync images(len(error_message), errmsg=error_message)

  !ERROR: An image-set that is an int-expr must be a scalar or a rank-one array
  sync images(invalid_rank)

  !ERROR: Must have INTEGER type, but is LOGICAL(4)
  sync images([1], stat=invalid_type)

  !ERROR: Must be a scalar value, but is a rank-1 array
  sync images(*, stat=non_scalar)

  !ERROR: Must have CHARACTER type, but is LOGICAL(4)
  sync images(1, errmsg=invalid_type)

  !ERROR: The stat-variable in a sync-stat-list may not be repeated
  sync images(1, stat=sync_status, stat=superfluous_stat)

  !ERROR: The stat-variable in a sync-stat-list may not be repeated
  sync images(1, stat=sync_status, errmsg=error_message, stat=superfluous_stat)

  !ERROR: The errmsg-variable in a sync-stat-list may not be repeated
  sync images([1], errmsg=error_message, errmsg=superfluous_errmsg)

  !ERROR: The errmsg-variable in a sync-stat-list may not be repeated
  sync images([1], stat=sync_status, errmsg=error_message, errmsg=superfluous_errmsg)

  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  sync images(*, stat=coindexed_integer[1])

  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  sync images(1, errmsg=coindexed_character[1])

end program test_sync_images
