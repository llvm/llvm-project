! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for errors in sync memory statements based on the
! statement specification in section 11.6.5 of the Fortran 2018 standard.

program test_sync_memory
  implicit none

  integer sync_status
  character(len=128) error_message

  !___ standard-conforming statements ___

  sync memory
  sync memory()
  sync memory(stat=sync_status)
  sync memory(                  errmsg=error_message)
  sync memory(stat=sync_status, errmsg=error_message)

  !___ non-standard-conforming statements ___

  !______ invalid sync-stat-lists: invalid stat= ____________

  !ERROR: expected execution part construct
  sync memory(status=sync_status)

  ! Invalid sync-stat-list: missing stat-variable
  !ERROR: expected execution part construct
  sync memory(stat)

  ! Invalid sync-stat-list: missing 'stat='
  !ERROR: expected execution part construct
  sync memory(sync_status)

  !______ invalid sync-stat-lists: invalid errmsg= ____________

  ! Invalid errmsg-variable keyword
  !ERROR: expected execution part construct
  sync memory(errormsg=error_message)

  ! Invalid sync-stat-list: missing 'errmsg='
  !ERROR: expected execution part construct
  sync memory(error_message)

  ! Invalid sync-stat-list: missing errmsg-variable
  !ERROR: expected execution part construct
  sync memory(errmsg)

end program test_sync_memory
