** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   SAVE statements.
	program p
	parameter(n = 11)
	integer rslts(n), expect(n)

	data expect / 12, 10,
     +                207, 308, 409, 10, 111, 14,
     +                17, 90, 18 /

c --- tests 1 - 2:  SAVE <null> statement.

	call sub1
	call sub1
	call sub2(1, rslts)

	rslts(2) = if3(6)

c --- tests 3 - 8 :  SAVE <list> statement.

	call sub4(rslts(3))
	call sub4(rslts(4))
	call sub4(rslts(5))

	call sub5(0, rslts(6))
	call sub5(100, rslts(7))
	call sub5(2, rslts(8))

c ---  tests 9 - 11:  more of same ...

	rslts(9) = if5(1)
	rslts(10) = if5(2)
	rslts(11) = if5(1)

c --- check results:

	call check(rslts, expect, n)
	end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine sub1()
	save
	common /c1/ i
	data j / 10 /
	j = j + 1
	i = j
	end

	subroutine sub2(k, r)
	common /c1/ i
	common /cxx/ x, y, z
	integer r(*), xx, yy(2)
c	external zz
	save
	r(k) = i
	end

	integer function if3(k)
c   --- save statement where there is nothing to save.
	save
	if3 = k
	if3 = if3 + 4
	end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine sub4(k)
	save i, j
	data i / 6 /
	if (i .eq. 6)  j = 100
	j = j + 100
	i = i + 1
	k = i + j
	end

	subroutine sub5(flag, ir)
c -- local var, i, which is not saved.
c -- .save. not data initialized.
c -- save common block before defined.

	integer flag
	save k, /cxx/
	common /cxx/ x, y, z

	i = flag
	if (flag .eq. 0) then
		k = 10
	else
		k = k + 1
	endif
	ir = i + k
	end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	integer function if5(i)
c --- save of array.
c --- non-saved cblock.
c --- non-empty .local. psect.
c --- test storage allocation/alignment within .save. psect.

	character*3 c1
	integer a(2)
	character*3 c2
	integer*2 k1, k2
	character*3 c3
	real*8 d
	character*3 c4
	common /comif5/ kk

	save k1, k2, d
	save c1, a, unref_var, c2, c3, d, c4

	data a /7, 80/, locv / 9 /
	
	a(i) = a(i) + 1
	if5 = a(i) + locv
	call dum(k1, k2, c1, c2, c3, c4, d)
	end

	subroutine dum(k1, k2, c1, c2, c3, c4, d)
	integer*2 k1, k2
	character*3 c1
	character*3 c2
	character*3 c3
	character*3 c4
	real*8 d
	end
