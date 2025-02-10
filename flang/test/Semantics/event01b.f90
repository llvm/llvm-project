! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in event post statements based on the
! statement specification in section 11.6.7 of the Fortran 2018 standard.
! Some of the errors in this test would be hidden by the errors in
! the test event01a.f90 if they were included in that file,
! and are thus tested here.

program test_event_post
  use iso_fortran_env, only : event_type
  implicit none

  ! event_type variables must be coarrays
  !ERROR: Variable 'non_coarray' with EVENT_TYPE or LOCK_TYPE must be a coarray
  type(event_type) non_coarray

  ! event_type potential object components must be nested in coarrays
  type :: has_event
    type(event_type) event
  end type
  type :: bad1
    type(has_event) component
  end type
  type :: bad2
    type(has_event), allocatable :: component
  end type
  type :: good1
    type(has_event), pointer :: component
  end type
  type :: good2
    type(has_event), allocatable :: component[:]
  end type
  !ERROR: Variable 'non_coarray_component1' with EVENT_TYPE or LOCK_TYPE potential component '%event' must be a coarray
  type(has_event) non_coarray_component1
  !ERROR: Variable 'non_coarray_component2' with EVENT_TYPE or LOCK_TYPE potential component '%component%event' must be a coarray
  type(bad1) non_coarray_component2
  !ERROR: Variable 'non_coarray_component3' with EVENT_TYPE or LOCK_TYPE potential component '%component%event' must be a coarray
  type(bad2) non_coarray_component3
  ! these are okay
  type(has_event) ok_non_coarray_component1[*]
  type(has_event), pointer :: ok_non_coarray_component2
  type(bad1) :: ok_non_coarray_component3[*]
  type(bad1), pointer :: ok_non_coarray_component4
  type(bad2) :: ok_non_coarray_component5[*]
  type(bad2), pointer :: ok_non_coarray_component6
  type(good1) ok_non_coarray_component7
  type(good2) ok_non_coarray_component8

  type(event_type) concert[*], occurrences(2)[*]
  integer non_event[*], sync_status, co_indexed_integer[*], superfluous_stat, non_scalar(1)
  character(len=128) error_message, co_indexed_character[*], superfluous_errmsg
  logical invalid_type

  !___ non-standard-conforming statements ___

  !______ invalid event-variable ____________________________

  ! event-variable must be event_type
  !ERROR: The event-variable must be of type EVENT_TYPE from module ISO_FORTRAN_ENV
  event post(non_event)

  !ERROR: Must be a scalar value, but is a rank-1 array
  event post(occurrences)

  !ERROR: Must be a scalar value, but is a rank-1 array
  event post(occurrences[1])

  !______ invalid sync-stat-lists: invalid stat= ____________

  !ERROR: Must have INTEGER type, but is LOGICAL(4)
  event post(concert, stat=invalid_type)

  !ERROR: Must be a scalar value, but is a rank-1 array
  event post(concert, stat=non_scalar)

  !______ invalid sync-stat-lists: invalid errmsg= ____________

  !ERROR: Must have CHARACTER type, but is LOGICAL(4)
  event post(concert, errmsg=invalid_type)

  !______ invalid sync-stat-lists: redundant sync-stat-list ____________

  !ERROR: The stat-variable in a sync-stat-list may not be repeated
  event post(concert, stat=sync_status, stat=superfluous_stat)

  !ERROR: The stat-variable in a sync-stat-list may not be repeated
  event post(concert, errmsg=error_message, stat=sync_status, stat=superfluous_stat)

  !ERROR: The stat-variable in a sync-stat-list may not be repeated
  event post(concert, stat=sync_status, errmsg=error_message, stat=superfluous_stat)

  !ERROR: The stat-variable in a sync-stat-list may not be repeated
  event post(concert, stat=sync_status, stat=superfluous_stat, errmsg=error_message)

  !ERROR: The errmsg-variable in a sync-stat-list may not be repeated
  event post(concert, errmsg=error_message, errmsg=superfluous_errmsg)

  !ERROR: The errmsg-variable in a sync-stat-list may not be repeated
  event post(concert, stat=sync_status, errmsg=error_message, errmsg=superfluous_errmsg)

  !ERROR: The errmsg-variable in a sync-stat-list may not be repeated
  event post(concert, errmsg=error_message, stat=sync_status, errmsg=superfluous_errmsg)

  !ERROR: The errmsg-variable in a sync-stat-list may not be repeated
  event post(concert, errmsg=error_message, errmsg=superfluous_errmsg, stat=sync_status)

  !______ invalid sync-stat-lists: coindexed stat-variable - C1173____________

  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  event post(concert, stat=co_indexed_integer[1])

  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  event post(concert, errmsg=co_indexed_character[1])

  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  event post(concert, stat=co_indexed_integer[1], errmsg=error_message)

  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  event post(concert, stat=sync_status, errmsg=co_indexed_character[1])

  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  event post(concert, stat=co_indexed_integer[1], errmsg=co_indexed_character[1])

  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  !ERROR: The stat-variable or errmsg-variable in a sync-stat-list may not be a coindexed object
  event post(concert, errmsg=co_indexed_character[1], stat=co_indexed_integer[1])

end program test_event_post
