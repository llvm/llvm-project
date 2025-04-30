** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*--- Induction initial values, update reaching defs
*    

	program p
	parameter (N=16)
	integer i, j
	double precision expect(N)
	double precision result(100)
	common /c/ result

	call cnstnt(4,result)

	data expect /
     +    1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 0.0,
     +    1.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 /

	call checkd(result, expect, N)
	end
C***********************************************************************
      SUBROUTINE CNSTNT(N,C)
      IMPLICIT DOUBLEPRECISION(A-H,O-Z)
      DIMENSION C(N,N)
C
C.....MOLECULAR CONSTANTS FOR WATER IN ANGSTROM, RADIAN, AND A.M.U.
C
      C(1,1)=1.0D0
      DO 1000 N1=2,N
	  NN=N1-1
	  TN=NN
	  C(1,N1)=1.0D0
	  TK=1.0D0
	  DO 1100 K1=2,N1
	      C(K1,NN)=C(K1-1,NN+1)*TN/TK
	      NN=NN-1
	      TN=TN-1.0D0
	      TK=TK+1.0D0
1100      CONTINUE
1000  CONTINUE
	return
	end
