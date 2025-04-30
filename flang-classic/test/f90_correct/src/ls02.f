** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Miscellaneous software pipelining bugs

	program test
	parameter (NUM=20)
	common result(NUM), expect(NUM)
	complex*16 result, expect
	complex*16 x(10),y(10)

	do 1 i = 1,10
	    call noswpipe
	    x(i) = cmplx(1d0*i,2d0*i)
	    y(i) = cmplx(3d0*i,5d0*i)
1	continue

	call zrot(5,x,1,y,2,(0.0d0),(-1.0d0))

	j = 1
	do 2 i = 1,10
	    call noswpipe
	    result(j) = x(i)
	    result(j+1) = y(i)
	    j = j + 2
c	    print *,x(i),y(i)
2	continue

	data expect /
     +  (-3.0, -5.0), (1.0, 2.0), (-9.0, -15.0), (6.0, 10.0),
     +  (-15.0, -25.0), (2.0, 4.0), (-21.0, -35.0), (12.0, 20.0),
     +  (-27.0, -45.0), (3.0, 6.0), (6.0, 12.0), (18.0, 30.0),
     +  (7.0, 14.0), (4.0, 8.0), (8.0, 16.0), (24.0, 40.0),
     +  (9.0, 18.0), (5.0, 10.0), (10.0, 20.0), (30.0, 50.0)
     +  /

	call checkd(result, expect, NUM*2)
	end

	subroutine noswpipe
	end

      SUBROUTINE zrot( N, X, INCX, Y, INCY, C, S )
      DOUBLE PRECISION   C, S
      INTEGER            INCX, INCY, N
      COMPLEX*16         X( * ), Y( * )
      DOUBLE PRECISION   ONE         , ZERO
      PARAMETER        ( ONE = 1.0D+0, ZERO = 0.0D+0 )
      COMPLEX*16         TEMP
      INTEGER            I, IX, IY
      IF( N.GT.0 )THEN
         IF( ( S.NE.ZERO ).OR.( C.NE.ONE ) )THEN
                  IF( INCY.GE.0 )THEN
                     IY = 1
                  ELSE
                     IY = 1 - ( N - 1 )*INCY
                  END IF
                  IF( INCX.GT.0 )THEN
                     DO 50, IX = 1, 1 + ( N - 1 )*INCX, INCX
                        TEMP    =  X( IX )
                        X( IX ) = -Y( IY )
                        Y( IY ) =  TEMP
                        IY      =  IY       + INCY
   50                CONTINUE
                  END IF
         END IF
      END IF
      RETURN
      END
