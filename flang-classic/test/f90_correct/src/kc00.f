** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Deletable store & IVDEP.

      program kc00
c
c     a very trivial fft comparer
c
      integer N
      parameter (N=32)
      doubleprecision res(N), expect(N)
      integer t
      data res /
     +  0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
     +  23,24,25,26,27,28,29,30,31
     + /
      data expect /
     +  0,1,30,31,28,29,26,27,24,25, 22,23,20,21,18,19,16,17,14,15,
     +  12,13,10,11,8,9,6,7,4,5,2,3    
     + /

      CALL fft (res, N/2)
      call checkd(res,expect, N)
      end
      SUBROUTINE FFT(A, n)
      IMPLICIT doubleprecision (A-H,O-Z)
      DIMENSION A(1)


      I1 = 3
      I2 = (n-1)*2 + 1
CDIR$ IVDEP
      DO 10 M=1,n/2
C     SWAP REAL AND IMAGINARY PORTIONS
d	print *, 'before'
d	print *, 'i1',i1
d	print *, 'i2',i2
d	write(*,99) i1, a(i1), i1+1, a(i1+1)
d	write(*,99) i2, a(i2), i2+1, a(i2+1)
d
      HREAL = A(I1)
      HIMAG = A(I1+1)
      A(I1) = A(I2)
      A(I1+1) = A(I2+1)
      A(I2) = HREAL
      A(I2+1) = HIMAG
d	print *, 'after'
d	write(*,99) i1, a(i1), i1+1, a(i1+1)
d	write(*,99) i2, a(i2), i2+1, a(i2+1)
      I1 = I1+2
      I2 = I2-2

   10 CONTINUE

99	format(' a(',i2,')=',f6.1) 
	end
