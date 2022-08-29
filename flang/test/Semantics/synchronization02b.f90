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

  ! Image set shall be a scalar or rank-1 array
  sync images(invalid_rank)

  !ERROR: Must have INTEGER type, but is LOGICAL(4)
  sync images([1], stat=invalid_type)

  !ERROR: Must be a scalar value, but is a rank-1 array
  sync images(*, stat=non_scalar)

  !ERROR: Must have CHARACTER type, but is LOGICAL(4)
  sync images(1, errmsg=invalid_type)

  ! No specifier shall appear more than once in a given sync-stat-list
  sync images(1, stat=sync_status, stat=superfluous_stat)

  ! No specifier shall appear more than once in a given sync-stat-list
  sync images([1], errmsg=error_message, errmsg=superfluous_errmsg)

  ! Fortran 2018 standard C1173: `stat` shall not be coindexed
  sync images(*, stat=coindexed_integer[1])

  ! Fortran 2018 standard C1173: `errmsg` shall not be coindexed
  sync images(1, errmsg=coindexed_character[1])

end program test_sync_images
