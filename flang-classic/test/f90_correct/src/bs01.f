** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Uses of integer*2 which are endian-dependent

	integer N
	parameter (N = 2)
	integer result(N), expect(N)
	integer*2 a(2)
	integer fun
	data a/1, 2/

	result(1) = fun(a(1) + a(2))

	result(2) = fun(iint(2))	! need special intrin (int2) ?


c ******* check results:

	call check(result, expect, N)

	data expect /
     +  3, 2
     +  /
	end
	integer function fun(arg)
	integer*2 arg	! addr of correct half of arg passed ?
	fun = arg
	end
