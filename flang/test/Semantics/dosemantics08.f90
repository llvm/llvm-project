! RUN: %S/test_errors.sh %s %t %f18
! C1138 -- 
! A branch (11.2) within a DO CONCURRENT construct shall not have a branch
! target that is outside the construct.

subroutine s1()
  do concurrent (i=1:10)
!ERROR: Control flow escapes from DO CONCURRENT
    goto 99
  end do

99 print *, "Hello"

end subroutine s1
