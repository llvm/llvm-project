! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks the acceptance of standard-conforming notify-wait-stmts based
! on the statement specification in section 11.6 of the Fortran 2023 standard.

program test_notify_wait
  use iso_fortran_env, only: notify_type
  implicit none

  type(notify_type) :: notify_var[*]
  integer :: count, count_array(1), sync_status, coindexed_integer[*]
  character(len=128) :: error_message

  !_______________________ standard-conforming statements ___________________________

  notify wait(notify_var)
  notify wait(notify_var, until_count=count)
  notify wait(notify_var, until_count=count_array(1))
  notify wait(notify_var, until_count=coindexed_integer[1])
  notify wait(notify_var, stat=sync_status)
  notify wait(notify_var, until_count=count, stat=sync_status)
  notify wait(notify_var, errmsg=error_message)
  notify wait(notify_var, until_count=count, errmsg=error_message)
  notify wait(notify_var, stat=sync_status, errmsg=error_message)
  notify wait(notify_var, until_count=count, stat=sync_status, errmsg=error_message)

end program test_notify_wait
