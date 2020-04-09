! RUN: %B/test/Semantics/test_errors.sh %s %flang %t
! Functions cannot use alt return

REAL FUNCTION altreturn01(X)
!ERROR: RETURN with expression is only allowed in SUBROUTINE subprogram
  RETURN 1
END
