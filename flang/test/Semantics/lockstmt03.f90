! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
! This test checks for semantic errors in lock statements based on the
! statement specification in section 11.6.10 of the Fortran 2018 standard.

program test_lock_stmt
  use iso_fortran_env, only: lock_type, event_type
  implicit none

  character(len=128) error_message, msg_array(10), coindexed_msg[*], repeated_msg
  integer status, stat_array(10), coindexed_int[*], non_bool, repeated_stat
  logical non_integer, bool, bool_array(10), non_char, coindexed_logical[*], repeated_bool
  type(lock_type) :: lock_var[*], lock_array(10)[*], non_coarray_lock
  type(event_type) :: not_lock_var[*]

  !___ non-standard-conforming statements ___

! type mismatches

  !ERROR: to be determined
  lock(not_lock_var)

  !ERROR: Must have LOGICAL type, but is INTEGER(4)
  lock(lock_var, acquired_lock=non_bool)

  !ERROR: Must have INTEGER type, but is LOGICAL(4)
  lock(lock_var, stat=non_integer)

  !ERROR: Must have CHARACTER type, but is LOGICAL(4)
  lock(lock_var, errmsg=non_char)

! rank mismatches

  !ERROR: Must be a scalar value, but is a rank-1 array
  lock(lock_array)

  !ERROR: Must be a scalar value, but is a rank-1 array
  lock(lock_var, acquired_lock=bool_array)

  !ERROR: Must be a scalar value, but is a rank-1 array
  lock(lock_var, stat=stat_array)

  !ERROR: Must be a scalar value, but is a rank-1 array
  lock(lock_var, errmsg=msg_array)

! corank mismatch

  !ERROR: to be determined
  lock(non_coarray_lock)

! C1173 - stat-variable and errmsg-variable shall not be a coindexed object

  !ERROR: to be determined
  lock(lock_var, stat=coindexed_int[1])

  !ERROR: to be determined
  lock(lock_var, errmsg=coindexed_msg[1])

  !ERROR: to be determined
  lock(lock_var, acquired_lock=coindexed_logical[1], stat=coindexed_int[1], errmsg=coindexed_msg[1])

! C1181 - No specifier shall appear more than once in a given lock-stat-list

  !ERROR: to be determined
  lock(lock_var, acquired_lock=bool, acquired_lock=repeated_bool)

  !ERROR: to be determined
  lock(lock_var, stat=status, stat=repeated_stat)

  !ERROR: to be determined
  lock(lock_var, errmsg=error_message, errmsg=repeated_msg)

  !ERROR: to be determined
  lock(lock_var, acquired_lock=bool, stat=status, errmsg=error_message, acquired_lock=repeated_bool)

  !ERROR: to be determined
  lock(lock_var, acquired_lock=bool, stat=status, errmsg=error_message, stat=repeated_stat)

  !ERROR: to be determined
  lock(lock_var, acquired_lock=bool, stat=status, errmsg=error_message, errmsg=repeated_msg)

  !ERROR: to be determined
  lock(lock_var, acquired_lock=bool, stat=status, errmsg=error_message, acquired_lock=repeated_bool, stat=repeated_stat)

  !ERROR: to be determined
  lock(lock_var, acquired_lock=bool, stat=status, errmsg=error_message, acquired_lock=repeated_bool, errmsg=repeated_msg)

  !ERROR: to be determined
  lock(lock_var, acquired_lock=bool, stat=status, errmsg=error_message, stat=repeated_stat, errmsg=repeated_msg)

  !ERROR: to be determined
  lock(lock_var, acquired_lock=bool, stat=status, errmsg=error_message, acquired_lock=repeated_bool, stat=repeated_stat, errmsg=repeated_msg)

end program test_lock_stmt
