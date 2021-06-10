! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! C1123 -- Expressions in DO CONCURRENT header cannot reference variables
! declared in the same header
PROGRAM dosemantics04
  IMPLICIT NONE
  INTEGER :: a, i, j, k, n

  !ERROR: DO CONCURRENT mask expression references variable 'n' in LOCAL locality-spec
  DO CONCURRENT (INTEGER *2 :: i = 1:10, i < j + n) LOCAL(n)
    PRINT *, "hello"
  END DO

  !ERROR: DO CONCURRENT mask expression references variable 'a' in LOCAL locality-spec
  DO 30 CONCURRENT (i = 1:n:1, j=1:n:2, k=1:n:3, a<3) LOCAL (a)
    PRINT *, "hello"
30 END DO

! Initial expression
  !ERROR: DO CONCURRENT limit expression may not reference index variable 'j'
  DO CONCURRENT (i = j:3, j=1:3)
  END DO

! Final expression
  !ERROR: DO CONCURRENT limit expression may not reference index variable 'j'
  DO CONCURRENT (i = 1:j, j=1:3)
  END DO

! Step expression
  !ERROR: DO CONCURRENT step expression may not reference index variable 'j'
  DO CONCURRENT (i = 1:3:j, j=1:3)
  END DO

  !ERROR: DO CONCURRENT limit expression may not reference index variable 'i'
  DO CONCURRENT (INTEGER*2 :: i = 1:3, j=i:3)
  END DO

END PROGRAM dosemantics04
