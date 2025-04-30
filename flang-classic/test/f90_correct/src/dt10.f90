!*** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!*** See https://llvm.org/LICENSE.txt for license information.
!*** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! array parameters in modules
	module md
	integer,dimension(3),parameter:: siz=(/ 10,20,30 /)
	end module
	program p
	use md
	integer,dimension(siz(1))::a
	integer,dimension(siz(2))::b
	integer,dimension(siz(3))::c
	parameter(n=6)
	integer result(n)
	integer expect(n)
	data expect/1,10,1,20,1,30/
	result(1) = lbound(a,1)
	result(2) = ubound(a,1)
	result(3) = lbound(b,1)
	result(4) = ubound(b,1)
	result(5) = lbound(c,1)
	result(6) = ubound(c,1)
	call check(result, expect, n)
	end
