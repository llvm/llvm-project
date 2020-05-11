! RUN: %S/test_errors.sh %s %t %f18
! Functions cannot use alt return

REAL FUNCTION altreturn01(X)
!ERROR: RETURN with expression is only allowed in SUBROUTINE subprogram
  RETURN 1
END
