! RUN: %python %S/test_errors.py %s %flang_fc1
!ERROR: IF statement is not allowed in IF statement
IF (A > 0.0) IF (B < 0.0) A = LOG (A)
END
