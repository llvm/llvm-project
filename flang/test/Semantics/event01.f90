! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in event post statements based on the
! statement specification in section 11.6.7 of the Fortran 2018 standard.

program test_event_post
  use iso_fortran_env, only : event_type
  implicit none

  ! event_type variables must be coarrays
  type(event_type) non_coarray

  type(event_type) concert[*], occurrences(2)[*]
  integer non_event[*], sync_status, co_indexed_integer[*], superfluous_stat, non_scalar(1)
  character(len=128) error_message, co_indexed_character[*], superfluous_errmsg
  logical invalid_type

  !___ standard-conforming statement ___

  event post(concert)
  event post(concert[1])
  event post(occurrences(1))
  event post(occurrences(1)[1])
  event post(concert, stat=sync_status)
  event post(concert,                   errmsg=error_message)
  event post(concert, stat=sync_status, errmsg=error_message)

  !___ non-standard-conforming statement ___

  !______ invalid event-variable ____________________________

  ! event-variable must be event_type
  event post(non_event)

  ! event-variable must be a coarray
  event post(non_coarray)

  ! event-variable must be a scalar variable
  event post(occurrences)

  ! event-variable must be a scalar variable
  event post(occurrences[1])

  ! event-variable has an unknown keyword argument
  !ERROR: expected ')'
  event post(event=concert)

  !______ invalid sync-stat-lists: invalid stat= ____________

  ! Invalid stat-variable keyword
  !ERROR: expected ')'
  event post(concert, status=sync_status)

  ! Stat-variable must an integer scalar
  event post(concert, stat=invalid_type)

  ! Stat-variable must an integer scalar
  event post(concert, stat=non_scalar)

  ! Invalid sync-stat-list: missing stat-variable
  !ERROR: expected ')'
  event post(concert, stat)

  ! Invalid sync-stat-list: missing 'stat='
  !ERROR: expected ')'
  event post(concert, sync_status)

  !______ invalid sync-stat-lists: invalid errmsg= ____________

  ! Invalid errmsg-variable keyword
  !ERROR: expected ')'
  event post(concert, errormsg=error_message)

  ! Invalid errmsg-variable argument typing
  event post(concert, errmsg=invalid_type)

  ! Invalid sync-stat-list: missing 'errmsg='
  !ERROR: expected ')'
  event post(concert, error_message)

  ! Invalid sync-stat-list: missing errmsg-variable
  !ERROR: expected ')'
  event post(concert, errmsg)

  !______ invalid event-variable: redundant event-variable ____________

  ! Too many arguments
  !ERROR: expected ')'
  event post(concert, occurrences(1))

  ! Too many arguments
  !ERROR: expected ')'
  event post(concert, occurrences(1), stat=sync_status, errmsg=error_message)

  !______ invalid sync-stat-lists: redundant sync-stat-list ____________

  ! No specifier shall appear more than once in a given sync-stat-list
  event post(concert, stat=sync_status, stat=superfluous_stat)

  ! No specifier shall appear more than once in a given sync-stat-list
  event post(concert, errmsg=error_message, errmsg=superfluous_errmsg)

  !______ invalid sync-stat-lists: coindexed stat-variable ____________

  ! Check constraint C1173 from the Fortran 2018 standard
  event post(concert, stat=co_indexed_integer[1])

  ! Check constraint C1173 from the Fortran 2018 standard
  event post(concert, errmsg=co_indexed_character[1])

end program test_event_post
