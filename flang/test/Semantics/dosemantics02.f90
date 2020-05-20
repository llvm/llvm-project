! RUN: %S/test_errors.sh %s %t %f18
! C1121 -- any procedure referenced in a concurrent header must be pure

! Also, check that the step expressions are not zero.  This is prohibited by
! Section 11.1.7.4.1, paragraph 1.

SUBROUTINE do_concurrent_c1121(i,n)
  IMPLICIT NONE
  INTEGER :: i, n, flag
  !ERROR: DO CONCURRENT mask expression may not reference impure procedure 'random'
  DO CONCURRENT (i = 1:n, random() < 3)
    flag = 3
  END DO

  CONTAINS
    IMPURE FUNCTION random() RESULT(i)
      INTEGER :: i
      i = 35
    END FUNCTION random
END SUBROUTINE do_concurrent_c1121

SUBROUTINE s1()
  INTEGER, PARAMETER :: constInt = 0

  ! Warn on this one for backwards compatibility
  DO 10 I = 1, 10, 0
  10 CONTINUE

  ! Warn on this one for backwards compatibility
  DO 20 I = 1, 10, 5 - 5
  20 CONTINUE

  ! Error, no compatibility requirement for DO CONCURRENT
  !ERROR: DO CONCURRENT step expression may not be zero
  DO CONCURRENT (I = 1 : 10 : 0)
  END DO

  ! Error, this time with an integer constant
  !ERROR: DO CONCURRENT step expression may not be zero
  DO CONCURRENT (I = 1 : 10 : constInt)
  END DO
end subroutine s1
