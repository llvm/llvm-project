! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell

subroutine x(n)
   call x1(n)
   if (n == 0) goto 88
   print*, 'x'
contains
   subroutine x1(n)
      if (n == 0) goto 77 ! ok
      print*, 'x1'
      !ERROR: Label '88' was not found
      goto 88
77 end subroutine x1
88 end
