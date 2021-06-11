! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Functions cannot use alt return

REAL FUNCTION altreturn01(X)
!ERROR: RETURN with expression is only allowed in SUBROUTINE subprogram
  RETURN 1
END
