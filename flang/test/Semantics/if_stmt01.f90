! RUN: %S/test_errors.sh %s %flang %t
! Simple check that if statements are ok.

IF (A > 0.0) A = LOG (A)
END
