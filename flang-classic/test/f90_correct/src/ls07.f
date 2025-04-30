** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   SW pipelining bug
      program ctest
      parameter (NN=5)
      INTEGER            INFO, LDB, N, NRHS
      REAL               D( NN )
      COMPLEX            B( NN, 1 ), E( NN )
      INTEGER            I, J
      logical upper
      complex expect(NN)
   
      e(1) = (1.0, 1)
      e(2) = (2.0, 2)
      e(3) = (3.0, 3)
      e(4) = (4.0, 4)
      e(5) = (5.0, 5)
      d(1) = 1
      d(2) = 1
      d(3) = 1
      d(4) = 1
      d(5) = 1

      nrhs = 1

      upper = nrhs .eq. n

      n = NN
      do 11 i = 2, 5
         b(i,1) = cmplx(i,i)
11    continue
      b(1,1) = (1,1)
      INFO = 0
*
      call cpttrs(upper,n,nrhs,d,e,b,5,info)
*
      data expect/ (1,1),(2,0),(-1,-1),(4,10),(29,-51) /
      call check(b, expect,NN*2)

*
      END

      SUBROUTINE CPTTRS( UPper, N, NRHS, D, E, B, LDB, INFO )
      INTEGER            INFO, LDB, N, NRHS
      REAL               D( * )
      COMPLEX            B( LDB, * ), E( * )
*     ..
      LOGICAL            UPPER
      INTEGER            I, J
d      INFO = 0
d      IF( UPPER ) THEN
d         DO 30 J = 1, NRHS
d            DO 10 I = 2, N
d               B( I, J ) = B( I, J ) - B( I-1, J )*CONJG( E( I-1 ) )
d   10       CONTINUE
d            B( N, J ) = B( N, J ) / D( N )
d            DO 20 I = N - 1, 1, -1
d               B( I, J ) = B( I, J ) / D( I ) - B( I+1, J )*E( I )
d   20       CONTINUE
d   30    CONTINUE
d      ELSE
d         DO 60 J = 1, NRHS
	    j = 1
            DO 40 I = 2, N
               B( I, J ) = B( I, J ) - B( I-1, J )*E( I-1 )
   40       CONTINUE
d            B( N, J ) = B( N, J ) / D( N )
d            DO 50 I = N - 1, 1, -1
d               B( I, J ) = B( I, J ) / D( I ) -
d     $                     B( I+1, J )*CONJG( E( I ) )
d   50       CONTINUE
d   60    CONTINUE
d      END IF
*
      RETURN
*
      END




