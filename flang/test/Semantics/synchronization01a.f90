! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for errors in sync all statements based on the
! statement specification in section 11.6.3 of the Fortran 2018 standard.

program test_sync_all
  implicit none

  integer sync_status
  character(len=128) error_message

  !___ standard-conforming statement ___

  sync all
  sync all()
  sync all(stat=sync_status)
  sync all(                  errmsg=error_message)
  sync all(stat=sync_status, errmsg=error_message)

  !___ non-standard-conforming statement ___

  !______ invalid sync-stat-lists: invalid stat= ____________

  !ERROR: expected execution part construct
  sync all(status=sync_status)

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

  ! Invalid sync-stat-list: missing 'errmsg='
  !ERROR: expected execution part construct
  sync all(error_message)

  ! Invalid sync-stat-list: missing errmsg-variable
  !ERROR: expected execution part construct
  sync all(errmsg)

end program test_sync_all
