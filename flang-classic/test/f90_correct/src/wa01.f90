!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!	check that index variables used in implied DO in array
!	constructor do not change variable of same name (i below)
!	but that those in implied DO in IO do change variable (j below)
	program pp
	integer i,j
	integer a(10)
	integer result(2),expect(2)
	data expect/1,11/
	i=1
	j = 2
	a=(/(i*2,i=1,10)/)
	write(*,*) (a(j),j=1,10)
	result(1) = i
	result(2) = j
	call check(result,expect,2)
	end
