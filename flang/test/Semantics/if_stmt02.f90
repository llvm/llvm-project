! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
!ERROR: IF statement is not allowed in IF statement
IF (A > 0.0) IF (B < 0.0) A = LOG (A)
END
