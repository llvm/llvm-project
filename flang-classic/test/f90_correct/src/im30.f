** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Troublesome statement functions

	program p
	parameter(N=1)
	common /exp/result, expect
	integer result(N)
	integer expect(N)

	data expect / 1 /

	result(1) = if1()

c  --- check the results:

	call check(result, expect, n)
	end

	integer function if1()
	common /fast/ a(100)
	real*8 a
	real*8 x1wxyz
	integer istuff, stuff
	locf(x1wxyz) = ishft(%loc(x1wxyz),-3)
	istuff = locf(a(1))
	stuff = ishft(%loc(a(1)),-3)
	if (istuff .ne. stuff) then
	    if1 = 99
	else
	    if1 = 1
	endif
	end
