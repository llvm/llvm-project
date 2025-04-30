** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

**---	0 format & character -- endian dependent

	parameter (N=3)
	integer result(N), expect(N)

	integer*4 endian
	integer*2 half(2)
	equivalence (endian, half)
	data endian /1/

	character *9 buf
	data buf /'030261143'/
	character *3 a

	read(buf,99) a
99	format(o9)

	result(2) = ichar(a(2:2))
	if (half(2) .eq. 1) then
c	-----  BIG ENDIAN
	    result(1) = ichar(a(1:1))
	    result(3) = ichar(a(3:3))
	else
c	-----  LITTLE ENDIAN
	    result(3) = ichar(a(1:1))
	    result(1) = ichar(a(3:3))
	endif

	data expect /'141'o, '142'o, '143'o/

	call check(result, expect, N)
	end
