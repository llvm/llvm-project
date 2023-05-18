! RUN: %python %S/test_errors.py %s %flang_fc1
! C1131 -- check valid and invalid DO loop naming

PROGRAM C1131
  IMPLICIT NONE
  ! Valid construct
  validDo: DO WHILE (.true.)
      PRINT *, "Hello"
    END DO ValidDo

  ! Missing name on END DO
  missingEndDo: DO WHILE (.true.)
      PRINT *, "Hello"
!ERROR: DO construct name required but missing
    END DO

  ! Missing name on DO
  DO WHILE (.true.)
      PRINT *, "Hello"
!ERROR: DO construct name unexpected
    END DO missingDO

END PROGRAM C1131
