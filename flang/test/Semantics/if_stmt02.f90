! RUN: %S/test_errors.sh %s %flang %t
!ERROR: IF statement is not allowed in IF statement
IF (A > 0.0) IF (B < 0.0) A = LOG (A)
END
