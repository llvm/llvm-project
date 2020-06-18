! RUN: %S/test_errors.sh %s %t %f18
! Simple check that if statements are ok.

IF (A > 0.0) A = LOG (A)
END
