** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Adjustable arrays.

	parameter(n = 14)
	common i1, i2, i3, i4, i5
	common /r/rslts(n)
	integer expect(n), rslts, ctoi
	real a(3,2)
	character h(2,1,1,1,1,1,2)
	complex cm(2:5, 2), c, cfunc

	ctoi(c) = real(c) + aimag(c)
	data i1, i2, i3, i4, i5 / 1, 2, 3, 4, 5 /

c --- tests 1 - 2:

	kl = 0
	ku = n - 1
	call sub1(kl, rslts, ku)

c --- tests 3 - 5:

	a(1,1) = 16.1
	a(3,2) = 2.0
	rslts(3) = ifunc(a)
	rslts(4) = a(2,1)
	rslts(5) = a(1,2)

c --- tests 6 - 11:

	call sub2(h, i1, cm)
	rslts(6) = ichar(h(1,1,1,1,1,1,1))
	rslts(7) = ichar(h(1,1,1,1,1,1,2))
	rslts(8) = ichar(h(2,1,1,1,1,1,1))
	rslts(9) = ichar(h(2,1,1,1,1,1,2))
	rslts(10) = ctoi( cm(2,1) )
	rslts(11) = ctoi( cm(3,2) )

c --- tests 12 - 14:

	rslts(12) = ctoi(cfunc(cm,2,5))
	rslts(13) = ctoi(cm(2,2))
	rslts(14) = ctoi(cm(3,2))

c --- check results:

	call check(rslts, expect, n)

	data expect / 7, 6,
     +                18, 2, -2,
     +                10, 12, 11, 12, 3, 7,
     +                11, 3, 70            /
	end

c --------------------------------------------------------- c

	subroutine sub1(lb, a, ub)
	integer lb, ub, a(lb:ub)

	data i1 / 1 /

	lb = 99
	ub = 99
	a(0) = 7
	a(i1) = 6
	end

c -------------------------------------------------------- c

	function ifunc(a)
	dimension a(i3, i4-2)
	common i1, i2, i3, i4, i5

	ifunc = a(1,1) + a(i3, i2)
	a(2,1) = 2.0
	a(i4-3, i2) = -2.0

	return
	end

c -------------------------------------------------------- c

	subroutine sub2(h, ii1, cm)
	character*1 h(2, ii1, 1, i3-2, i4/4, i2+(-i1), ii1*2)
	complex cm(i1**2+1 : i5, ii1 + ii1)

	common i1, i2, i3, i4, i5

	h(1,1,1,1,1,1,1) = char(10)
	h(i2,1,1,1,1,i1,1) = char(11)
	h(1,1,1,1,1,1,i1+1) = char(12)
	h(2,1,i1,1,1,1,2) = h(i1,1,1,1,1,1,2)

	cm(2,1) = (1.0, 2.0)
	cm(i3, 2) = (3.0, 4.0)

	end

c -------------------------------------------------------- c

	complex function cfunc(a, n, m)
	complex a(n:m, n)

	call sub3(a, n, a(n,2))
	cfunc = a(4, 2)
	end

c -------------------------------------------------------- c

	subroutine sub3(a, n, c)
	complex c, a
	dimension a(n:5, n)

	c = (1.0, 2.0)
	a(3, 2) = (3.0, 67.0)
	a(4, 2) = (5.0, 6.0)
	end
