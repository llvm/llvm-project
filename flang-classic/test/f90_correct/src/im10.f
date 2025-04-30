** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Statement functions with arguments (character statement functions
*   and character arguments not tested).

	i(n) = 1 + n
	call mysub(i(3), i(-5))
	end

	subroutine mysub(i4, in4)
	parameter(n = 31)
	integer TRUE
	parameter(TRUE = -1)
	integer rslts(n), expect(n)
	logical lrslts(n)
	equivalence(lrslts, rslts)

	common /if/ i7
	complex cc, cf1, cf2, cf3
	integer ctoi
	double precision df1, df2, dd, df3, df4
	logical and, l1, l2, lf1, lf2, lf3, lf4

c --- define the statement functions:

	if(i4) = i4 - 2
	if2(a, b) = a + b
	if3(ichar) = ichar + ichar + ichar
	if4(i7) = i7 * in4 + if3(i7)

	xf(a, b, c, d, e) = a + b - c + e - d
	xf2(a, i, j) = a
	data i7, x2 / 7, 2.0 /
	xf3(a, b) = b
	xf4(cc) = aimag(cc)
	xf5(a, b) = xadd(b, 2.3 + a + b)
	cos(x) = max(x, 5.0)

	df1(i, j) = dble(i) * j
	df2(dd) = i7
	df3(dd) = dd
	df4(a) = 2.0 / a

	cf1(i, j) = cmplx(i, j)
	cf2(cc) = (i4 - 2) * (cc + cc)
	cf3(x, sin) = cmplx( xadd(x,sin), xadd(x,-sin) )
	ctoi(cc) = int(real(cc) + aimag(cc))

	and(l1, l2) = l1 .and. l2
	lf1(l1) = .not. l1
	lf2(cc) = real(cc) .gt. 0.0  .or.  aimag(cc) .gt. 0
	lf3(i) = and( and(i .gt. 0, .true.), and(i.ge.0, i.gt. -2) )
	lf4(l1, l2) = and(l1, l2)

c --- tests 1 - 4:     INTEGER statement functions:

	rslts(1) = if(5)
	rslts(2) = if(if(if(i7))) * 5
	rslts(if2(x2, 1.1))
     +     = if2( float(if(i7+1)), real(if2(4.0, i7+1.0)))
	rslts(4) = if3(i4) + if4(93+i7)

c --- tests 5 - 11:    REAL statement functions:

	rslts(5) = xf(10.0, 11.6, float(i4-2), -100.0, 1.6)
	rslts(6) = xf2(2.1, 7, i4)
	rslts(7) = nint(xf3(2.8, -10.55))
	rslts(8) = xf4(cmplx(i4, in4))
	rslts(9) = xadd( xf5(1.1, 2.0), 100.0)
	rslts(10) = xf5(2.0, xadd(10.0, 5.0))
	rslts(11) = cos(-6.0) * cos(real(i7))

c --- tests 12 - 16:   DOUBLE PRECISION statement functions:

	rslts(12) = max(df1(i4, in4), -17.0d0)
	rslts(13) = df2(100d0) * 2.2
	rslts(14) = df3( df2( df3(1d1) ) )
	rslts(15) = df3( dble( if3( int( xadd(4.4, 6.6)))))
	rslts(16) = df4( xadd(4e1, 2e1) ) * 10000

c --- tests 17 - 19:   COMPLEX statement functions:

	rslts(17) = aimag( cf1(i4, in4))
	rslts(18) = ctoi( cf2( cf1(6, 7) ) )
	rslts(19) = cf3(1.0, 3.0) + aimag(cf3(1.0, 4.0))

c --- tests 20 - 26:   LOGICAL statement functions:

	lrslts(20) = and(.true., .false.)
	lrslts(21) = lf1(i4 .lt. in4)
	lrslts(22) = lf3(i4 + in4)
	lrslts(23) = lf3(1)
	lrslts(24) = lf4(i4 .gt. 0, x2 .gt. 1.1)
	data cc / (1.0, 2.0) /
	lrslts(25) = lf2( cc )
	lrslts(26) = lf2( - cc )

c --- tests 27 - 31:   LOGICAL statement functions in IF conditions:

	do 10 i = if(29), 31
10    		rslts(i) = 3
	if (and(i4 .gt. 0, in4.lt.0))   rslts(27) = 1
	if (.not. lf1( i4 .eq. in4))    rslts(28) = 1
	if (x2 .eq. 2.0  .and.  lf2( (-1.0, 1.0) ))  rslts(29) = 1
	if (lf3(i4) .and. lf3(in4))   rslts(30) = 1
	if (lf4(.true., lf1(lf1(i4 .ne. 0))))  rslts(31) =  1
	
c --- check results:

	call check(rslts, expect, n)

	data expect / 3, 5, 18, -88,
     +                121, 2, -11, -4, 107, 34, 35,
     +                -16, 15, 7, 33, 333,
     +                -4, 52, 1,
     +                0, TRUE, 0, TRUE, TRUE, TRUE, 0,
     +                1, 3, 1, 3, 1                 /
	end

	function xadd(a, b)
	xadd = a + b
	end

	function dadd(a, b)
	real*8 dadd, a, b
	dadd = a + b
	return
	end
