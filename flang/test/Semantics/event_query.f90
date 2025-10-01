! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in event_query() subroutine based on the
! statement specification in section 16.9.72 of the Fortran 2018 standard.

program test_event_query
  use iso_fortran_env, only : event_type
  implicit none(type,external)

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
  ! ERROR: Actual argument for 'event=' has bad type 'INTEGER(4)'
  call event_query(non_event, counter)

  ! event-variable must be a scalar variable
  ! ERROR: 'event=' argument has unacceptable rank 1
  call event_query(occurrences, counter)

  ! event-variable must not be coindexed
  ! ERROR: EVENT= argument to EVENT_QUERY must not be coindexed
  call event_query(concert[1], counter)

  ! event-variable has an unknown keyword argument
  ! ERROR: unknown keyword argument to intrinsic 'event_query'
  call event_query(events=concert, count=counter)

  ! event-variable has an argument mismatch
  ! ERROR: Actual argument for 'event=' has bad type 'INTEGER(4)'
  call event_query(event=non_event, count=counter)

  ! count must be an integer
  ! ERROR: Actual argument for 'count=' has bad type 'LOGICAL(4)'
  call event_query(concert, non_integer)

  ! count must be an integer scalar
  ! ERROR: 'count=' argument has unacceptable rank 1
  call event_query(concert, non_scalar)

  ! count must be have a decimal exponent range
  ! no smaller than that of default integer
  ! ERROR: COUNT= argument to EVENT_QUERY must be an integer with kind >= 4
  call event_query(concert, non_default)

  ! count is an intent(out) argument
  ! ERROR: Actual argument associated with INTENT(OUT) dummy argument 'count=' is not definable
  ! ERROR: '4_4' is not a variable or pointer
  call event_query(concert, 4)

  ! count has an unknown keyword argument
  ! ERROR: unknown keyword argument to intrinsic 'event_query'
  call event_query(concert, counts=counter)

  ! count has an argument mismatch
  ! ERROR: Actual argument for 'count=' has bad type 'LOGICAL(4)'
  call event_query(concert, count=non_integer)

  ! stat must be an integer
  ! ERROR: Actual argument for 'stat=' has bad type 'LOGICAL(4)'
  call event_query(concert, counter, non_integer)

  ! stat must be an integer scalar
  ! ERROR: 'stat=' argument has unacceptable rank 1
  call event_query(concert, counter, non_scalar)

  ! stat is an intent(out) argument
  ! ERROR: Actual argument associated with INTENT(OUT) dummy argument 'stat=' is not definable
  ! ERROR: '8_4' is not a variable or pointer
  call event_query(concert, counter, 8)

  ! stat has an unknown keyword argument
  ! ERROR: unknown keyword argument to intrinsic 'event_query'
  call event_query(concert, counter, status=sync_status)

  ! stat has an argument mismatch
  ! ERROR: Actual argument for 'stat=' has bad type 'LOGICAL(4)'
  call event_query(concert, counter, stat=non_integer)

  ! stat must not be coindexed
  ! ERROR: 'stat' argument to 'event_query' may not be a coindexed object
  call event_query(concert, counter, coindexed[1])

  ! Too many arguments
  ! ERROR: too many actual arguments for intrinsic 'event_query'
  call event_query(concert, counter, sync_status, array(1))

  ! Repeated event keyword
  ! ERROR: repeated keyword argument to intrinsic 'event_query'
  call event_query(event=concert, event=occurrences(1), count=counter)

  ! Repeated count keyword
  ! ERROR: repeated keyword argument to intrinsic 'event_query'
  call event_query(event=concert, count=counter, count=array(1))

  ! Repeated stat keyword
  ! ERROR: repeated keyword argument to intrinsic 'event_query'
  call event_query(event=concert, count=counter, stat=sync_status, stat=array(1))

end program test_event_query
