! Check for various alt return error conditions

       SUBROUTINE TEST (N, *, *)
       REAL :: R
       COMPLEX :: Z
       INTEGER, DIMENSION(2) :: B
       IF ( N .EQ. 0 ) RETURN
       IF ( N .EQ. 1 ) RETURN 1
       IF ( N .EQ. 2 ) RETURN 2
       IF ( N .EQ. 3 ) RETURN 3
       IF ( N .EQ. 3 ) RETURN N
       IF ( N .EQ. 3 ) RETURN N * N
       IF ( N .EQ. 3 ) RETURN B(N)
       IF ( N .EQ. 3 ) RETURN B
       IF ( N .EQ. 3 ) RETURN R
       IF ( N .EQ. 3 ) RETURN Z
       RETURN 2
       END
