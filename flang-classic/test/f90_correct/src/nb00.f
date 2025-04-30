** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

**---  Z edit descriptor for double precision (run-time is endian dependent)

	parameter(N=16)
	integer result(N)
	integer expect(N)
	character*16 buf
	double precision d
	data d /1.0/    ! 3FF0000000000000

	write(buf, 99) d
99	format(z16)
	do i = 1, N
	    result(i) = ichar(buf(i:i))
	enddo

	data expect/'33'x, '46'x, '46'x, 13*'30'x/
	call check(result, expect, N)
	end
