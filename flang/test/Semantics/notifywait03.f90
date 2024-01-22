! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in notify wait statements based on the
! statement specification in section 11.6 of the Fortran 2023 standard.
! Some of the errors in this test would be hidden by the errors in
! the test notify02.f90 if they were included in that file,
! and are thus tested here.

program test_notify_wait
  use iso_fortran_env, only : notify_type
  implicit none

  ! notify_type variables must be coarrays
  type(notify_type) :: non_coarray

  type(notify_type) :: notify_var[*], notify_array(2)[*]
  integer :: count, count_array(1), non_notify[*], sync_status, coindexed_integer[*], superfluous_stat, non_scalar(1)
  character(len=128) :: error_message, non_scalar_char(1), coindexed_character[*], superfluous_errmsg
  logical :: invalid_type

  !____________________ non-standard-conforming statements __________________________

  !_________________________ invalid notify-variable ________________________________

  !ERROR: The notify-variable must be of type NOTIFY_TYPE from module ISO_FORTRAN_ENV
  notify wait(non_notify)

  !ERROR: The notify-variable must be a coarray
  notify wait(non_coarray)

  !ERROR: A notify-variable in a NOTIFY WAIT statement may not be a coindexed object
  notify wait(notify_var[1])

  !ERROR: A notify-variable in a NOTIFY WAIT statement may not be a coindexed object
  notify wait(notify_array(1)[1])

  !ERROR: Must be a scalar value, but is a rank-1 array
  notify wait(notify_array)

  !_____________ invalid event-wait-spec-lists: invalid until-spec _________________

  !ERROR: Must have INTEGER type, but is LOGICAL(4)
  notify wait(notify_var, until_count=invalid_type)

  !ERROR: Must be a scalar value, but is a rank-1 array
  notify wait(notify_var, until_count=non_scalar)

  !_________________ invalid sync-stat-lists: invalid stat= ________________________

  !ERROR: Must have INTEGER type, but is LOGICAL(4)
  notify wait(notify_var, stat=invalid_type)

  !ERROR: Must be a scalar value, but is a rank-1 array
  notify wait(notify_var, stat=non_scalar)

  !________________ invalid sync-stat-lists: invalid errmsg= _______________________

  !ERROR: Must have CHARACTER type, but is LOGICAL(4)
  notify wait(notify_var, errmsg=invalid_type)

  !ERROR: Must be a scalar value, but is a rank-1 array
  notify wait(notify_var, errmsg=non_scalar_char)

  !______ invalid event-wait-spec-lists: redundant event-wait-spec-list ____________

  !ERROR: Until-spec in a event-wait-spec-list may not be repeated
  notify wait(notify_var, until_count=count, until_count=count_array(1))

  !ERROR: Until-spec in a event-wait-spec-list may not be repeated
  notify wait(notify_var, until_count=count, stat=sync_status, until_count=count_array(1))

  !ERROR: Until-spec in a event-wait-spec-list may not be repeated
  notify wait(notify_var, until_count=count, errmsg=error_message, until_count=count_array(1))

  !ERROR: Until-spec in a event-wait-spec-list may not be repeated
  notify wait(notify_var, until_count=count, stat=sync_status, errmsg=error_message, until_count=count_array(1))

  !ERROR: A stat-variable in a event-wait-spec-list may not be repeated
  notify wait(notify_var, stat=sync_status, stat=superfluous_stat)

  !ERROR: A stat-variable in a event-wait-spec-list may not be repeated
  notify wait(notify_var, stat=sync_status, until_count=count, stat=superfluous_stat)

  !ERROR: A stat-variable in a event-wait-spec-list may not be repeated
  notify wait(notify_var, stat=sync_status, errmsg=error_message, stat=superfluous_stat)

  !ERROR: A stat-variable in a event-wait-spec-list may not be repeated
  notify wait(notify_var, stat=sync_status, until_count=count, errmsg=error_message, stat=superfluous_stat)

  !ERROR: A errmsg-variable in a event-wait-spec-list may not be repeated
  notify wait(notify_var, errmsg=error_message, errmsg=superfluous_errmsg)

  !ERROR: A errmsg-variable in a event-wait-spec-list may not be repeated
  notify wait(notify_var, errmsg=error_message, until_count=count, errmsg=superfluous_errmsg)

  !ERROR: A errmsg-variable in a event-wait-spec-list may not be repeated
  notify wait(notify_var, errmsg=error_message, stat=superfluous_stat, errmsg=superfluous_errmsg)

  !ERROR: A errmsg-variable in a event-wait-spec-list may not be repeated
  notify wait(notify_var, errmsg=error_message, until_count=count, stat=superfluous_stat, errmsg=superfluous_errmsg)

  !_____________ invalid sync-stat-lists: coindexed stat-variable - C1173 __________________

  !ERROR: The stat-variable or errmsg-variable in a event-wait-spec-list may not be a coindexed object
  notify wait(notify_var, stat=coindexed_integer[1])

  !ERROR: The stat-variable or errmsg-variable in a event-wait-spec-list may not be a coindexed object
  notify wait(notify_var, errmsg=coindexed_character[1])

  !ERROR: The stat-variable or errmsg-variable in a event-wait-spec-list may not be a coindexed object
  notify wait(notify_var, stat=coindexed_integer[1], errmsg=error_message)

  !ERROR: The stat-variable or errmsg-variable in a event-wait-spec-list may not be a coindexed object
  notify wait(notify_var, stat=sync_status, errmsg=coindexed_character[1])

  !ERROR: The stat-variable or errmsg-variable in a event-wait-spec-list may not be a coindexed object
  !ERROR: The stat-variable or errmsg-variable in a event-wait-spec-list may not be a coindexed object
  notify wait(notify_var, stat=coindexed_integer[1], errmsg=coindexed_character[1])

  !ERROR: The stat-variable or errmsg-variable in a event-wait-spec-list may not be a coindexed object
  !ERROR: The stat-variable or errmsg-variable in a event-wait-spec-list may not be a coindexed object
  notify wait(notify_var, errmsg=coindexed_character[1], stat=coindexed_integer[1])

end program test_notify_wait
