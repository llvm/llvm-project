** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Optimizer bug in loop processing

	program ka02
	integer expect(209), np(209)

	data expect/
     + 15, -5, -20, -10, -25, -1, -16, -6, -21, -11,
     + -26, -2, -17, -7, -22, -12, -27, -3, -18, -8,
     + -23, -13, -28, -4, -19, -9, -24, -14, -29, 180*0
     + /

	data np/
     + 15, 5, 20, 10, 25, 1, 16, 6, 21, 11,
     + 26, 2, 17, 7, 22, 12, 27, 3, 18, 8,
     + 23, 13, 28, 4, 19, 9, 24, 14, 29, 180*0
     + /

	j = 0
	nn = 29
	goto 914

910   continue
      K=KK
      KK=NP(K)
      NP(K)=-KK
      IF(KK .NE. J) GO TO 910
      K3=KK
914   J=J+1
      KK=NP(J)
      IF(KK .LT. 0) GO TO 914
      IF(KK .NE. J) GO TO 910
      NP(J)=-J
      IF(J .NE. NN) GO TO 914


	call check(np, expect, 29)
	end
