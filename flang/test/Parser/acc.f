! RUN: %flang_fc1 -fsyntax-only -fopenacc %s 2>&1
C Test file for OpenACC directives in fixed-form Fortran
      PROGRAM ACCTEST
      IMPLICIT NONE
      INTEGER :: N, I, J
      PARAMETER (N=100)
      REAL :: A(N), B(N), C(N), D(N)
      REAL :: SUM

C Initialize arrays
      DO I = 1, N
         A(I) = I * 1.0
         B(I) = I * 2.0
         C(I) = 0.0
         D(I) = 1.0
      END DO

C Basic data construct using C$ACC
C$ACC DATA COPYIN(A,B) COPYOUT(C)
      DO I = 1, N
         C(I) = A(I) + B(I)
      END DO
C$ACC END DATA

* Parallel construct with loop using *$ACC
*$ACC PARALLEL PRESENT(A,B,C)
*$ACC LOOP
      DO I = 1, N
         C(I) = C(I) * 2.0
      END DO
*$ACC END PARALLEL

C Nested loops with collapse - C$ACC style
C$ACC PARALLEL LOOP COLLAPSE(2)
      DO I = 1, N
         DO J = 1, N
            A(J) = A(J) + B(J)
         END DO
      END DO
C$ACC END PARALLEL LOOP

* Combined parallel loop with reduction - *$ACC style
      SUM = 0.0
*$ACC PARALLEL LOOP REDUCTION(+:SUM)
      DO I = 1, N
         SUM = SUM + C(I)
      END DO
*$ACC END PARALLEL LOOP

C Kernels construct - C$ACC with continuation
C$ACC KERNELS 
C$ACC+ COPYOUT(A)
      DO I = 1, N
         A(I) = A(I) * 2.0
      END DO
C$ACC END KERNELS

* Data construct with update - *$ACC with continuation
*$ACC DATA COPY(B)
*$ACC+ PRESENT(D)
      B(1) = 999.0
*$ACC UPDATE HOST(B(1:1))
      PRINT *, 'B(1) = ', B(1)
*$ACC END DATA

C Mixed style directives in nested constructs
C$ACC DATA COPY(A,B,C)
*$ACC PARALLEL LOOP
      DO I = 1, N
         A(I) = B(I) + C(I)
      END DO
*$ACC END PARALLEL LOOP
C$ACC END DATA

* Subroutine call within data region - *$ACC style
*$ACC DATA COPY(A,B,C)
      CALL SUB1(A, B, C, N)
*$ACC END DATA

      PRINT *, 'Sum = ', SUM
      END PROGRAM

C Subroutine with mixed ACC directive styles
      SUBROUTINE SUB1(X, Y, Z, M)
      INTEGER M, I
      REAL X(M), Y(M), Z(M)

*$ACC PARALLEL PRESENT(X,Y)
C$ACC LOOP PRIVATE(I)
      DO I = 1, M
         Z(I) = X(I) + Y(I)
      END DO
C$ACC END LOOP
*$ACC END PARALLEL
      RETURN
      END SUBROUTINE 