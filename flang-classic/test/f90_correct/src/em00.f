** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   BLOCKDATA subprograms.

c  -- items tested include:
c     (1) named and unnamed block datas.
c     (2) initialization of common block in multiple block datas (NO)
c     (3) initialization of multiple cblocks in 1 blockdata.
c     (4) all allowed types of statement in a blockdata.
c     (5) local variables in a blockdata.
c     (6) common block name same as block data name.
c     (7) initialization of blank common in a blockdata.
c     (8) initialization of a big common block.
c     (9) empty block data.

	block data x
	character*3 c
	common /c/ c
	common /b/ i(55000)

	data (i(j), j = 1, 55000, 55000-1) / 7, 77 /
	data c(1:1) /'c'/
	data c(2:2) /'b'/
	data c(3:4-1) / 'a' /
	end

	program p
	common /c/ c
	character*3 c
	common ivar
	common /xx2/ x, y
	common /b/ i(55000)

	integer rslts(8), expect(8)

	rslts(1) = ichar(c(1:1))
	rslts(2) = ichar(c(2:2))
	rslts(3) = ichar(c(3:3))
	rslts(4) = ivar
	rslts(5) = x
	rslts(6) = y
	rslts(7) = i(1)
	rslts(8) = i(55000)

	call check(rslts, expect, 8)
	data expect / '143'o, '142'o, '141'o, 11, 11, 12, 7, 77 /
	end

	BLOCKDATA 
	implicit character*3 (c)
	character a
	parameter(a = 'a')
	common /c/ c
	integer i
	save i
	equivalence (i, j), (k, kk)

	data i / 33 /, k / 44 /
	end

	blockdata xx
	common blank
	parameter (xxx = 6.99 + 5.1)
	common /xx2/ x
	dimension x(2)
	equivalence (ivar, blank)
	common /c/ c
	character*3 c

	data (x(i-1), i = 3, 3, 2) / xxx /
	data ivar / 11 /, x(1) / 11.1/
	end

	blockdata xxx
	end
