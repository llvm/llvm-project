! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
! This test checks for semantic errors in event_query() subroutine based on the
! statement specification in section 16.9.72 of the Fortran 2018 standard.

program test_event_query
  use iso_fortran_env, only : event_type
  implicit none

  ! event_type variables must be coarrays
  type(event_type) non_coarray

  type(event_type) concert[*], occurrences(2)[*]
  integer non_event[*], counter, array(1), coarray[*], sync_status, coindexed[*], non_scalar(1)
  integer(kind=1) non_default
  logical non_integer

  !___ standard-conforming calls with required arguments _______

  call event_query(concert, counter)
  call event_query(occurrences(1), counter)
  call event_query(concert, array(1))
  call event_query(concert, coarray[1])
  call event_query(event=concert, count=counter)
  call event_query(count=counter, event=concert)

  !___ standard-conforming calls with all arguments ____________
  call event_query(concert, counter, sync_status)
  call event_query(concert, counter, array(1))
  call event_query(event=concert, count=counter, stat=sync_status)
  call event_query(stat=sync_status, count=counter, event=concert)

  !___ non-standard-conforming calls _______

  ! event-variable must be event_type
  call event_query(non_event, counter)

  ! event-variable must be a coarray
  call event_query(non_coarray, counter)

  ! event-variable must be a scalar variable
  call event_query(occurrences, counter)

  ! event-variable must not be coindexed
  call event_query(concert[1], counter)

  ! event-variable has an unknown keyword argument
  call event_query(events=concert, count=counter)

  ! event-variable has an argument mismatch
  call event_query(event=non_event, count=counter)

  ! count must be an integer
  call event_query(concert, non_integer)

  ! count must be an integer scalar
  call event_query(concert, non_scalar)

  ! count must be have a decimal exponent range
  ! no smaller than that of default integer
  call event_query(concert, non_default)

  ! count is an intent(out) argument
  call event_query(concert, 4)

  ! count has an unknown keyword argument
  call event_query(concert, counts=counter)

  ! count has an argument mismatch
  call event_query(concert, count=non_integer)

  ! stat must be an integer
  call event_query(concert, counter, non_integer)

  ! stat must be an integer scalar
  call event_query(concert, counter, non_scalar)

  ! stat is an intent(out) argument
  call event_query(concert, counter, 8)

  ! stat has an unknown keyword argument
  call event_query(concert, counter, status=sync_status)

  ! stat has an argument mismatch
  call event_query(concert, counter, stat=non_integer)

  ! stat must not be coindexed
  call event_query(concert, counter, coindexed[1])

  ! Too many arguments
  call event_query(concert, counter, sync_status, array(1))

  ! Repeated event keyword
  call event_query(event=concert, event=occurrences(1), count=counter)

  ! Repeated count keyword
  call event_query(event=concert, count=counter, count=array(1))

  ! Repeated stat keyword
  call event_query(event=concert, count=counter, stat=sync_status, stat=array(1))

end program test_event_query
