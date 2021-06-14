! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Simple check that if statements are ok.

IF (A > 0.0) A = LOG (A)
END
