! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for various alt return error conditions

       SUBROUTINE TEST (N, *, *)
       REAL :: R
       COMPLEX :: Z
       INTEGER, DIMENSION(2) :: B
       IF ( N .EQ. 0 ) RETURN
       IF ( N .EQ. 1 ) RETURN 1
       IF ( N .EQ. 2 ) RETURN 2
       IF ( N .EQ. 3 ) RETURN 3
       IF ( N .EQ. 4 ) RETURN N
       IF ( N .EQ. 5 ) RETURN N * N
       IF ( N .EQ. 6 ) RETURN B(N)
       !ERROR: Must be a scalar value, but is a rank-1 array
       IF ( N .EQ. 7 ) RETURN B
       !ERROR: Must have INTEGER type, but is REAL(4)
       IF ( N .EQ. 8 ) RETURN R
       !ERROR: Must have INTEGER type, but is COMPLEX(4)
       IF ( N .EQ. 9 ) RETURN Z
       RETURN 2
       END
