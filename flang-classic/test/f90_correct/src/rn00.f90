!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!*
!* Tests that the random number generator does not return zeros
!* This occurs for all releases before 5.1-6 when using -r8

PROGRAM P
IMPLICIT NONE  

! - - - local variables - - -
!
integer, parameter :: N=3
integer, parameter :: NT=1
REAL*8 :: pool(N,N,N) ! random number pool
INTEGER :: i
INTEGER, dimension(NT) :: exp, res


CALL RANDOM_SEED()               ! set seed to random number based on time
CALL RANDOM_NUMBER(pool)        ! fill pool with random data ( 0. -> 1. )

exp(1) = 3
res(1) = 0
DO i = 1,N
   IF (pool(i,i,i) > 0) THEN
      res(1) = res(1) + 1
   ELSE
      WRITE (*,*) 'Random numbers should not be zeros'
      WRITE (*,*) '---- pool(1:3,1:3,i) i=', i
      WRITE (*,*) pool(1:3,1:3,i)
   ENDIF
END DO

call check(res, exp, NT)

END PROGRAM P


