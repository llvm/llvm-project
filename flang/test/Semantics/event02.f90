! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in event wait statements based on the
! statement specification in section 11.6.8 of the Fortran 2018 standard.

program test_event_wait
  use iso_fortran_env, only : event_type
  implicit none

  ! event_type variables must be coarrays
  type(event_type) non_coarray

  type(event_type) concert[*], occurrences(2)[*]
  integer threshold, indexed(1), non_event[*], sync_status, co_indexed_integer[*], superfluous_stat, non_scalar(1)
  character(len=128) error_message, non_scalar_char(1), co_indexed_character[*], superfluous_errmsg
  logical invalid_type

  !_______________________ standard-conforming statement ___________________________

  event wait(concert)
  event wait(occurrences(1))
  event wait(concert, until_count=threshold)
  event wait(concert, until_count=indexed(1))
  event wait(concert, until_count=co_indexed_integer[1])
  event wait(concert,                        stat=sync_status)
  event wait(concert, until_count=threshold, stat=sync_status)
  event wait(concert,                                          errmsg=error_message)
  event wait(concert, until_count=threshold,                   errmsg=error_message)
  event wait(concert,                        stat=sync_status, errmsg=error_message)
  event wait(concert, until_count=threshold, stat=sync_status, errmsg=error_message)

  !____________________ non-standard-conforming statement __________________________

  !_________________________ invalid event-variable ________________________________

  ! event-variable must be event_type
  event wait(non_event)

  ! event-variable must be a coarray
  event wait(non_coarray)

  ! event-variable must not be coindexed
  event wait(concert[1])

  ! event-variable must not be coindexed
  event wait(occurrences(1)[1])

  ! event-variable must be a scalar variable
  event wait(occurrences)

  ! event-variable has an unknown expression
  !ERROR: expected ')'
  event wait(event=concert)

  !_____________ invalid event-wait-spec-lists: invalid until-spec _________________

  ! Invalid until-spec keyword
  !ERROR: expected ')'
  event wait(concert, until_amount=threshold)

  ! Until-spec must be an integer scalar
  event wait(concert, until_count=invalid_type)

  ! Until-spec must be an integer scalar
  event wait(concert, until_count=non_scalar)

  ! Invalid until-spec: missing until-spec variable
  !ERROR: expected ')'
  event wait(concert, until_count)

  ! Invalid until-spec: missing 'until_count='
  !ERROR: expected ')'
  event wait(concert, threshold)

  !_________________ invalid sync-stat-lists: invalid stat= ________________________

  ! Invalid stat-variable keyword
  !ERROR: expected ')'
  event wait(concert, status=sync_status)

  ! Stat-variable must be an integer scalar
  event wait(concert, stat=invalid_type)

  ! Stat-variable must be an integer scalar
  event wait(concert, stat=non_scalar)

  ! Invalid sync-stat-list: missing stat-variable
  !ERROR: expected ')'
  event wait(concert, stat)

  ! Invalid sync-stat-list: missing 'stat='
  !ERROR: expected ')'
  event wait(concert, sync_status)

  !________________ invalid sync-stat-lists: invalid errmsg= _______________________

  ! Invalid errmsg-variable keyword
  !ERROR: expected ')'
  event wait(concert, errormsg=error_message)

  ! Errmsg-variable must be a character scalar
  event wait(concert, errmsg=invalid_type)

  ! Errmsg-variable must be a character scalar
  event wait(concert, errmsg=non_scalar_char)

  ! Invalid sync-stat-list: missing 'errmsg='
  !ERROR: expected ')'
  event wait(concert, error_message)

  ! Invalid sync-stat-list: missing errmsg-variable
  !ERROR: expected ')'
  event wait(concert, errmsg)

  !______________ invalid event-variable: redundant event-variable _________________

  !ERROR: expected ')'
  event wait(concert, occurrences(1))

  !ERROR: expected ')'
  event wait(concert, occurrences(1), stat=sync_status, errmsg=error_message)

  !______ invalid event-wait-spec-lists: redundant event-wait-spec-list ____________

  ! No specifier shall appear more than once in a given event-wait-spec-list
  event wait(concert, until_count=threshold, until_count=indexed(1))

  ! No specifier shall appear more than once in a given event-wait-spec-list
  event wait(concert, stat=sync_status, stat=superfluous_stat)

  ! No specifier shall appear more than once in a given event-wait-spec-list
  event wait(concert, errmsg=error_message, errmsg=superfluous_errmsg)

  !_____________ invalid sync-stat-lists: coindexed stat-variable __________________

  ! Check constraint C1173 from the Fortran 2018 standard
  event wait(concert, stat=co_indexed_integer[1])

  ! Check constraint C1173 from the Fortran 2018 standard
  event wait(concert, errmsg=co_indexed_character[1])

end program test_event_wait
