! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in lock statements based on the
! statement specification in section 11.6.10 of the Fortran 2018 standard.

program test_lock_stmt
  use iso_fortran_env, only: lock_type
  implicit none

  character(len=128) error_message
  integer status
  logical bool
  type(lock_type) :: lock_var[*]

  !___ non-standard-conforming statements ___

! missing required lock-variable

  !ERROR: expected '('
  lock

  !ERROR: expected '='
  lock()

  !ERROR: expected ')'
  lock(acquired_lock=bool)

  !ERROR: expected ')'
  lock(stat=status)

  !ERROR: expected ')'
  lock(errmsg=error_message)

! specifiers in lock-stat-list are not variables

  !ERROR: expected ')'
  lock(lock_var, acquired_lock=.true.)

  !ERROR: expected ')'
  lock(lock_var, stat=1)

  !ERROR: expected ')'
  lock(lock_var, errmsg='c')

! specifier typos

  !ERROR: expected ')'
  lock(lock_var, acquiredlock=bool, stat=status, errmsg=error_message)

  !ERROR: expected ')'
  lock(lock_var, acquired_lock=bool, status=status, errmsg=error_message)

  !ERROR: expected ')'
  lock(lock_var, acquired_lock=bool, stat=status, errormsg=error_message)

end program test_lock_stmt
