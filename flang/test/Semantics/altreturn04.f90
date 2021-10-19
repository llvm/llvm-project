! RUN: %python %S/test_errors.py %s %flang_fc1
! Functions cannot use alt return

REAL FUNCTION altreturn01(X)
!ERROR: RETURN with expression is only allowed in SUBROUTINE subprogram
  RETURN 1
END
