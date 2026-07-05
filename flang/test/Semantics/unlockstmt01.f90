! RUN: %python %S/test_errors.py %s %flang_fc1
program test_unlock_stmt

  use iso_fortran_env, only: lock_type

  type(LOCK_TYPE) :: myLock[*], locks(10)[*]
  integer :: stat_variable
  character(len = 128) :: errmsg_variable

  !___ standard-conforming statements ___
  UNLOCK(myLock)
  UNLOCK(locks(3))
  Unlock(locks(1), stat = stat_variable)
  Unlock(locks(2), ERRMSG = errmsg_variable)
  Unlock(locks(4), stat = stat_variable, ERRMSG = errmsg_variable)
end program test_unlock_stmt
