** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*--- Induction uses, nested loops, creating bla uses
*    

	program p
	parameter (N=3)
	integer i, j, result(N), expect(N)
	common i, j, result, expect

	data expect /
     +   6, 			! t1
     +   10, 			! t2
     +   4			! t3
     +  /

	call t1(result(1), 4, 1)

	call t2(result(2), 4, 1)

	result(3) = 0
	call t3(result(3), 2, 2)

	call check(result, expect, N)
	end
	subroutine t1(ir, n, is)
	dimension ir(1)
	ir(is) = 0
	do j = 1, n
	    jm1 = j - 1
	    do k = is, jm1	! bla count is jm1 - is
		ir(is) = ir(is) + 1
	    enddo
	enddo
	end
	subroutine t2(ir, n, is)
	dimension ir(1)
	ir(is) = 0
	do j = 1, n
	    do k = is, j	! bla count is j - is
		ir(is) = ir(is) + 1
	    enddo
	enddo
	end
	subroutine t3(ires,ni, nj)
* bug due to register allocator running out of registers before
* a register can be assigned to the bla; bla requires registers!
* bug manifests itself as an internal complier error.
      common/jif/ jif(1000), imax, kmk, kmkd, kmku, kmq, kmr

      DO 112 I = 1,NI
            J = 2
            IJ = I + JIF(J)
           IJK = IJ + KMK
           IJR = IJ + KMR
           IJQ = IJ + KMQ
          IJKE = IJK + 1
          IJKW = IJK - 1
          IJKN = IJK + IMAX
          IJKS = IJK - IMAX
          IJKD = IJ + KMKD
          IJKU = IJ + KMKU
            IJ = I + JIF(J)
           IJK = IJ + KMK
           IJR = IJ + KMR
           IJQ = IJ + KMQ
          IJKE = IJK + 1
          IJKW = IJK - 1
          IJKN = IJK + IMAX
	  ires = ires + 1
  112 CONTINUE
      DO 114 J = 1,NJ
	  I = 2
            IJ = I + JIF(J)
           IJK = IJ + KMK
           IJR = IJ + KMR
           IJQ = IJ + KMQ
          IJKE = IJK + 1
          IJKW = IJK - 1
          IJKN = IJK + IMAX
          IJKS = IJK - IMAX
          IJKD = IJ + KMKD
          IJKU = IJ + KMKU
	  ires = ires + 1
  114 CONTINUE
      RETURN
      END
