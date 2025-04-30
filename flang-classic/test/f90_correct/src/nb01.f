** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

**---  z format - complex array
	parameter(N=4)
	integer result(N), expect(N)
	complex c(2)
	character*16 buf(2)
	data buf /'3ff00000345abcde', 'abcdef987654321f'/
	read(buf,99) c
99	format(2z8)
	read(buf(1)(1:8),  100) result(1)
	read(buf(1)(9:16), 100) result(2)
	read(buf(2)(1:8),  100) result(3)
	read(buf(2)(9:16), 100) result(4)
100	format(z8)

	data expect /'3ff00000'x, '345abcde'x, 'abcdef98'x, '7654321f'x/
	call check(result, expect, N)
	end
