** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

* Vectorizer - zero strides & streaming data
      parameter (N=4)
      double precision result(N), expect(N)
      double precision a(2), b(2)
      common /dat/a, b
      equivalence(result(1), a(1))

      a(1) = 1.0
      a(2) = 2.0
      b(1) = 3.0
      b(2) = 4.0

      call dswap(2, a, 0, b, 0)   ! swaps the first elements 2 times => nochange

      data expect/1.0, 2.0, 3.0, 4.0/
	call checkd(result, expect, N)
      end


      subroutine dswap (n,x,incx,y,incy)
      integer n, incx, incy
      double precision x(*), y(*)
c
c     Swap vectors.
c     y <==> x
c
      integer i, ix, iy
      double precision t
c
   20 ix = 1
      iy = 1
	  do i = 1, n
	     t = y(iy)
	     y(iy) = x(ix)
	     x(ix) = t
	     ix = ix + incx
	     iy = iy + incy
	  enddo
	  return
      return
      end
