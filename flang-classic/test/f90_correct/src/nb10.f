** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

**---   O format, double precision

	parameter(N=22)
	integer result(N), expect(N)
	data result/N*99/
	character*22 buf
	double precision d
	data d /'0400441616155074102142'o/
	write(buf, 99)d
99	format(o22)
	do i = 1, N
	    result(i) = ichar(buf(i:i)) - ichar('0')
	enddo

	data expect /
     +  -16, 4, 0, 0, 4, 4, 1, 6, 1, 6, 1, 5,
     +  5, 0, 7, 4, 1, 0, 2, 1, 4, 2 /
	call check(result, expect, N)
	end
