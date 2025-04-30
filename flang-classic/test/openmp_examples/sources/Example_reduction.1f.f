! @@name:	reduction.1f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
SUBROUTINE REDUCTION1(A, B, C, D, X, Y, N)
    REAL :: X(*), A, D
    INTEGER :: Y(*), N, B, C
    INTEGER :: I
    A = 0
    B = 0
    C = Y(1)
    D = X(1)
    !$OMP PARALLEL DO PRIVATE(I) SHARED(X, Y, N) REDUCTION(+:A) &
    !$OMP& REDUCTION(IEOR:B) REDUCTION(MIN:C)  REDUCTION(MAX:D)
      DO I=1,N
        A = A + X(I)
        B = IEOR(B, Y(I))
        C = MIN(C, Y(I))
        IF (D < X(I)) D = X(I)
      END DO

END SUBROUTINE REDUCTION1
