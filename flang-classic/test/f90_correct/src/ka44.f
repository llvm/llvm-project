** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*--- Induction last value bugs
*    

	program ka44
	parameter (N=1)
	integer i, j, result(N), expect(N)
	common i, j, result, expect

	data expect /21/
	call pas4f(10, 0, result(1))
	call check(result, expect, N)
	end
      SUBROUTINE PAS4F (LA, nd4, ires)
C
      INTEGER*4    LA
      INTEGER*4    K,L
      INTEGER*4    I2,J2,ND4
         J2 = 1
         DO 40 K = 0, nd4

            DO 30 L = 1, LA
               J2 = J2 + 2
   30       CONTINUE
	    ires = j2	! bug - last val incr considered invariant
   40    CONTINUE
      RETURN
      END
