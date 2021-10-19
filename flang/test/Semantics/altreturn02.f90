! RUN: %python %S/test_errors.py %s %flang_fc1
! Check subroutine with alt return

       SUBROUTINE TEST (N, *, *)
       IF ( N .EQ. 0 ) RETURN
       IF ( N .EQ. 1 ) RETURN 1
       RETURN 2
       END
