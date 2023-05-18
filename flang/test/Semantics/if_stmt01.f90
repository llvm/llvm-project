! RUN: %python %S/test_errors.py %s %flang_fc1
! Simple check that if statements are ok.

IF (A > 0.0) A = LOG (A)
END
