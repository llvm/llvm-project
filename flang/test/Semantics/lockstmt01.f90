! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in lock statements based on the
! statement specification in section 11.6.10 of the Fortran 2018 standard.

program test_lock_stmt
  use iso_fortran_env, only: lock_type
  implicit none

  character(len=128) error_message, msg_array(10)
  integer status, stat_array(10)
  logical bool, bool_array, coindexed_logical[*]
  type(lock_type) :: lock_var[*]

  !___ standard-conforming statements ___

  lock(lock_var)
  lock(lock_var[1])
  lock(lock_var[this_image() + 1])
  lock(lock_var, acquired_lock=bool)
  lock(lock_var, acquired_lock=bool_array(1))
  lock(lock_var, acquired_lock=coindexed_logical[1])
  lock(lock_var, stat=status)
  lock(lock_var, stat=stat_array(1))
  lock(lock_var, errmsg=error_message)
  lock(lock_var, errmsg=msg_array(1))

  lock(lock_var, stat=status, errmsg=error_message)
  lock(lock_var, errmsg=error_message, stat=status)
  lock(lock_var, acquired_lock=bool, stat=status)
  lock(lock_var, stat=status, acquired_lock=bool)
  lock(lock_var, acquired_lock=bool, errmsg=error_message)
  lock(lock_var, errmsg=error_message, acquired_lock=bool)

  lock(lock_var, acquired_lock=bool, stat=status, errmsg=error_message)
  lock(lock_var, acquired_lock=bool, errmsg=error_message, stat=status)
  lock(lock_var, stat=status, acquired_lock=bool, errmsg=error_message)
  lock(lock_var, stat=status, errmsg=error_message, acquired_lock=bool)
  lock(lock_var, errmsg=error_message, acquired_lock=bool, stat=status)
  lock(lock_var, errmsg=error_message, stat=status, acquired_lock=bool)

end program test_lock_stmt
