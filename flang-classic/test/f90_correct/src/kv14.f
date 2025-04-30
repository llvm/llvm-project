** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

* Vectorizer - argonne loop s422
*

cpgi$g novector
      program kv14
      integer ld, nloops
      parameter (ld=100,nloops=135)  	!PGI, ld=1000
      real dtime, ctime
      double precision array
      integer ntimes
      common /cdata/ array(1000*1000)	!PGI, ld*ld
      double precision a(ld),b(ld),c(ld),d(ld),e(ld),aa(ld,ld),
     +                 bb(ld,ld),cc(ld,ld)

      n      = 10
      ntimes = 100000

      call s422 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)

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

      subroutine chk (chksum,totit,n,t2,name)
c
c --  called by each loop to record and report results
c --  chksum is the computed checksum
c --  totit is the number of times the loop was executed
c --  n  is the length of the loop
c --  t2 is the time to execute loop 'name'
c
      integer nloops, nvl, totit, n
      double precision epslon, chksum, rnorm
      parameter (nloops=135,nvl=3,epslon=1.d-10)
      character*5 name
      double precision res
      integer ifail, expect

      ifail = 1
      res = 1.6982700302343163d0
      rnorm = sqrt((res-chksum)*(res-chksum))/chksum
      if ( ( rnorm .gt. epslon) .or. ( rnorm .lt. -epslon) ) then
d        write(*,98)name,n,chksum,res,rnorm,i
	 ifail = 1
      else
d        write(*,99)name,n,chksum,res,i
	 ifail = 0
      endif

d98    format(a6,i5,5x,'no time',2x,1p,e22.16,1x,1p,e22.16,1p
d     +,d13.4,9x,i3)
d99    format(a6,i5,5x,'no time',2x,1p,e22.16,1x,1p,e22.16,22x,i3)

      data expect /0/
      call check(ifail, expect, 1)

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

      subroutine dummy(ld,n,a,b,c,d,e,aa,bb,cc,s)
c
c --  called in each loop to make all computations appear required
c
      integer ld, n
      double precision a(n), b(n), c(n), d(n), e(n), aa(ld,n),
     +                 bb(ld,n), cc(ld,n)
      double precision s
      return
      end

      subroutine init(ld,n,a,b,c,d,e,aa,bb,cc,name)
      double precision zero, small, half, one, two, any, array
      parameter(any=0.0d0,zero=0.0d0,half=.5d0,one=1.0d0,
     +          two=2.0d0,small=.000001d0)
      integer unit, frac, frac2, ld, n
      parameter(unit=1, frac=-1, frac2=-2)
      double precision a(n), b(n), c(n), d(n), e(n), aa(ld,n),
     +                 bb(ld,n), cc(ld,n)
      common /cdata/ array(1000*1000)
      character*5 name

         call set1d(n,array,one,unit)
         call set1d(n,  a, any,frac2)

      end
cpgi$g vector
      subroutine s422 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     storage classes and equivalencing
c     common and equivalence statement
c     anti-dependence, threshold of 4
c
      integer ntimes, ld, n, i, nl, nn, vl
      parameter(nn=1000,vl=64)
      double precision a(n), b(n), c(n), d(n), e(n), aa(ld,n),
     +                 bb(ld,n), cc(ld,n)
      double precision x(nn), array
      double precision chksum, cs1d
      equivalence (x(1),array(5))
      real t2, ctime, dtime
      common /cdata/ array(1000000)

      call set1d(n,x,0.0d0,1)
      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s422 ')
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         x(i) = array(i+8) + a(i)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.d0)
   1  continue
      chksum = cs1d(n,x)
      call chk (chksum,ntimes*(n-8),n,t2,'s422 ')
      return
      end
