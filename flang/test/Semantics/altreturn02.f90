! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Check subroutine with alt return

       SUBROUTINE TEST (N, *, *)
       IF ( N .EQ. 0 ) RETURN
       IF ( N .EQ. 1 ) RETURN 1
       RETURN 2
       END
