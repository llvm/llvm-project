** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

* Vectorizer - stripmine, incorrect placement of temp stores of variant
*              expressions
*              iscbugs 911004a.f -- 911004f.f

	program p
	parameter (N=512)
	parameter (M=N/2)

	double precision x(N), y(N)
	double precision expect(N)

	data x /N*1.0/
	data y /N*2.0/
	data (expect(i),i=1,N,2) / M*2.0 /
	data (expect(i),i=2,N,2) / M*1.0 /

	incx = 2
	incy = -2
	call dswap (N/2,x,incx,y,incy)
	call checkd(x, expect, N)
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
      if (incx .lt. 0) ix = (-n+1)*incx + 1
      if (incy .lt. 0) iy = (-n+1)*incy + 1
c
c  need temps for computing the initial array addresses for x & y
c  these temps should be stored prior to the stripmine loop
c
      do 30 i = 1, n
         t = y(iy)
         y(iy) = x(ix)
         x(ix) = t
         ix = ix + incx
         iy = iy + incy
   30 continue
      return
      end
