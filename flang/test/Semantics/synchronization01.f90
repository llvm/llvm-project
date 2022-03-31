! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
! This test checks for semantic errors in sync all statements based on the
! statement specification in section 11.6.3 of the Fortran 2018 standard.

program test_sync_all
  implicit none

  integer sync_status, co_indexed_integer[*], superfluous_stat, non_scalar(1)
  character(len=128) error_message, co_indexed_character[*], superfluous_errmsg
  logical invalid_type
  
  !___ standard-conforming statement ___

  sync all
  sync all(stat=sync_status)
  sync all(                  errmsg=error_message)
  sync all(stat=sync_status, errmsg=error_message)
 
  !___ non-standard-conforming statement ___

  !______ invalid sync-stat-lists: invalid stat= ____________

  !ERROR: expected execution part construct
  sync all(status=sync_status)

  ! Stat-variable must an integer scalar
  !ERROR: TBD
  sync all(stat=invalid_type)

  ! Stat-variable must an integer scalar
  !ERROR: TBD
  sync all(stat=non_scalar)

  ! Invalid sync-stat-list: missing stat-variable
  !ERROR: expected execution part construct
  sync all(stat)

  ! Invalid sync-stat-list: missing 'stat='
  !ERROR: expected execution part construct
  sync all(sync_status)

  !______ invalid sync-stat-lists: invalid errmsg= ____________

  ! Invalid errmsg-variable keyword
  !ERROR: expected execution part construct
  sync all(errormsg=error_message)

  !ERROR: TBD
  sync all(errmsg=invalid_type)

  ! Invalid sync-stat-list: missing 'errmsg='
  !ERROR: expected execution part construct
  sync all(error_message)

  ! Invalid sync-stat-list: missing errmsg-variable
  !ERROR: expected execution part construct
  sync all(errmsg)

  !______ invalid sync-stat-lists: redundant sync-stat-list ____________

  ! No specifier shall appear more than once in a given sync-stat-list
  !ERROR: to be determined
  sync all(stat=sync_status, stat=superfluous_stat)

  ! No specifier shall appear more than once in a given sync-stat-list
  !ERROR: to be determined
  sync all(errmsg=error_message, errmsg=superfluous_errmsg)
 
  !______ invalid sync-stat-lists: coindexed stat-variable ____________

  ! Check constraint C1173 from the Fortran 2018 standard
  !ERROR: to be determined
  sync all(stat=co_indexed_integer[1])
 
  ! Check constraint C1173 from the Fortran 2018 standard
  !ERROR: to be determined
  sync all(errmsg=co_indexed_character[1])

end program test_sync_all
