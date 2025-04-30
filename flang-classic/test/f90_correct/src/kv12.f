** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

* Vectorizer - argonne loop s243
*

cpgi$g novector
      program kv12
      integer ld
      parameter (ld=100)
      integer n,ntimes
      double precision a(ld),b(ld),c(ld),d(ld),e(ld)

      n      = 10
      ntimes = 100000

      call s243 (ntimes,ld,n,ctime,dtime,a,b,c,d,e)

      end
      subroutine set1d(n,array,value,stride)
c
c  -- initialize one-dimensional arrays
c
      integer i, n, stride, frac, frac2
      double precision array(n), value
      parameter(frac=-1,frac2=-2)
      if ( stride .eq. frac ) then
         do 10 i=1,n
            array(i) = 1.0d0/dble(i)
10       continue
      elseif ( stride .eq. frac2 ) then
         do 15 i=1,n
            array(i) = 1.0d0/dble(i*i)
15       continue
      else
         do 20 i=1,n,stride
            array(i) = value
20       continue
      endif
      return
      end

      subroutine chk (chksum)

      double precision epslon, chksum, rnorm
      parameter (epslon=1.d-10)
      double precision res
      integer ifail, iexp
      parameter (res = 703168.96900448343d0)
      rnorm = sqrt((res-chksum)*(res-chksum))/chksum
      if ( ( rnorm .gt. epslon) .or. ( rnorm .lt. -epslon) ) then
d          print *, 'expected=', res, ' computed=', chksum
	  ifail = 1
      else
	  ifail = 0
      endif

      data iexp/0/

      call check(ifail, iexp, 1)
      return
      end

      double precision function cs1d(n,a)
c
c --  calculate one-dimensional checksum
c
      integer i,n
      double precision a(n), sum
      sum = 0.0d0
      do 10 i = 1,n
         sum = sum + a(i)
10    continue
      cs1d = sum
      return
      end

      subroutine dummy(ld,n,a,b,c,d,e,s)
c
c --  called in each loop to make all computations appear required
c
      integer ld, n
      double precision a(n), b(n), c(n), d(n), e(n)
      double precision s
      return
      end

      subroutine init(ld,n,a,b,c,d,e)
      double precision zero, small, half, one, two, any, array
      parameter(any=0.0d0,zero=0.0d0,half=.5d0,one=1.0d0,
     +          two=2.0d0,small=.000001d0)
      integer unit, frac, frac2, ld, n
      parameter(unit=1, frac=-1, frac2=-2)
      double precision a(n), b(n), c(n), d(n), e(n)

         call set1d(n,  a,zero,unit)
         call set1d(n,  b, one,unit)
         call set1d(n,  c, any,frac)
         call set1d(n,  d, any,frac)
         call set1d(n,  e, any,frac)

      return
      end
cpgi$g vector
      subroutine s243 (ntimes,ld,n,ctime,dtime,a,b,c,d,e)
c
c     node splitting
c     false dependence cycle breaking
c
      integer ntimes, ld, n, i, nl
      double precision a(n), b(n), c(n), d(n), e(n)
      double precision chksum, cs1d

      call init(ld,n,a,b,c,d,e)
      do 1 nl = 1,ntimes
      do 10 i = 1,n-1
         a(i) = b(i) + c(i)   * d(i)
         b(i) = a(i) + d(i)   * e(i)
         a(i) = b(i) + a(i+1) * d(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,1.d0)
  1   continue
      chksum = cs1d(n,a) + cs1d(n,b)
      call chk (chksum)
      return
      end
