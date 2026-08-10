! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in notify wait statements based on the
! statement specification in section 11.6 of the Fortran 2023 standard

program test_notify_wait
  use iso_fortran_env, only: notify_type
  implicit none

  ! notify_type variables must be coarrays
  type(notify_type) :: non_coarray

  type(notify_type) :: notify_var[*], redundant_notify[*]
  integer :: count, sync_status
  character(len=128) :: error_message

  !____________________ non-standard-conforming statements __________________________

  !_________________________ invalid notify-variable ________________________________

  ! notify-variable has an unknown expression
  !ERROR: expected '('
  notify wait(notify=notify_var)

  !_____________ invalid event-wait-spec-lists: invalid until-spec _________________

  ! Invalid until-spec keyword
  !ERROR: expected '('
  notify wait(notify_var, until_amount=count)

  ! Invalid until-spec: missing until-spec variable
  !ERROR: expected '('
  notify wait(notify_var, until_count)

  ! Invalid until-spec: missing 'until_count='
  !ERROR: expected '('
  notify wait(notify_var, count)

  !_________________ invalid sync-stat-lists: invalid stat= ________________________

  ! Invalid stat-variable keyword
  !ERROR: expected '('
  notify wait(notify_var, status=sync_status)

  ! Invalid sync-stat-list: missing stat-variable
  !ERROR: expected '('
  notify wait(notify_var, stat)

  ! Invalid sync-stat-list: missing 'stat='
  !ERROR: expected '('
  notify wait(notify_var, sync_status)

  !________________ invalid sync-stat-lists: invalid errmsg= _______________________

  ! Invalid errmsg-variable keyword
  !ERROR: expected '('
  notify wait(notify_var, errormsg=error_message)

  ! Invalid sync-stat-list: missing 'errmsg='
  !ERROR: expected '('
  notify wait(notify_var, error_message)

  ! Invalid sync-stat-list: missing errmsg-variable
  !ERROR: expected '('
  notify wait(notify_var, errmsg)

  !______________ invalid notify-variable: redundant notify-variable _________________

  !ERROR: expected '('
  notify wait(notify_var, redundant_notify)

  !ERROR: expected '('
  notify wait(notify_var, redundant_notify, stat=sync_status, errmsg=error_message)

end program test_notify_wait
