! RUN: %python %S/test_errors.py %s %flang_fc1
program testcriticalconstruct
  integer :: status
  character(len=128) :: errormessage
  integer :: total = 0
  integer :: i

  CRITICAL
  END CRITICAL

  Testname: Critical
  End critical Testname

  stat_variable: critical (STAT=status)
  End critical stat_variable

  errmsg_variable: critical (ERRMSG=errormessage)
  End critical errmsg_variable

  critical (ERRMSG=errormessage, STAT=status)
  End critical

  critical ()
  end critical

  critical (STAT=status)
    do i = 1, this_image()
        total = total + 1
    end do
    print *, "Total is: ", total
  End critical

  critical
    10 continue
    GO TO 10
  End critical

end program testcriticalconstruct
