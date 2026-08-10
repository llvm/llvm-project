! RUN: %python %S/test_errors.py %s %flang_fc1
! C1134 A CYCLE statement must be within a DO construct
!
! C1166 An EXIT statement must be within a DO construct

subroutine s1()
! this one's OK
  do i = 1,10
    cycle
  end do

! this one's OK
  do i = 1,10
    exit
  end do

! all of these are OK
  outer: do i = 1,10
    cycle
    inner: do j = 1,10
      cycle
    end do inner
    cycle
  end do outer

!ERROR: No matching DO construct for CYCLE statement
  cycle

!ERROR: No matching construct for EXIT statement
  exit

!ERROR: No matching DO construct for CYCLE statement
  if(.true.) cycle

!ERROR: No matching construct for EXIT statement
  if(.true.) exit

end subroutine s1
