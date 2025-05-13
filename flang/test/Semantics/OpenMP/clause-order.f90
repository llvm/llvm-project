! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! Ensure that checks on more than one data-sharing clause do not depend upon
! the clause order

PROGRAM main
  INTEGER:: I, N1, N2

  !ERROR: 'n1' appears in more than one data-sharing clause on the same OpenMP directive
  !$OMP PARALLEL DO PRIVATE(N1) SHARED(N1)
  DO I=1, 4
  ENDDO
  !$OMP END PARALLEL DO

  !ERROR: 'n2' appears in more than one data-sharing clause on the same OpenMP directive
  !$OMP PARALLEL DO SHARED(N2) PRIVATE(N2)
  DO I=1, 4
  ENDDO
  !$OMP END PARALLEL DO
END PROGRAM
