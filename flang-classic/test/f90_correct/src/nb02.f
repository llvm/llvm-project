** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

**---	z format & character -- endian dependent

	parameter (N=3)
	integer result(N), expect(N)

	integer*4 endian
	integer*2 half(2)
	equivalence (endian, half)
	data endian /1/

	character*6 buf
	data buf /'616263'/

	character*3 a

	read(buf,99) a
99	format(z6)

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

	data expect /'61'x, '62'x, '63'x/

	call check(result, expect, N)
	end
