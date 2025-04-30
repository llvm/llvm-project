!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!	Test that array constructors are allowed as 'put' argument to
!	random_seed

	program pp
	integer n
	integer, allocatable :: s(:)
	integer result,expect
	data expect/1/

	call random_seed(size=n)

	allocate (s(1:n))
	s = (/ ( i, i=1,n ) /)
	call random_seed(put=s)

	call random_seed(put=(/ ( i, i=1,n ) /) )

	result = 1
	call check(result,expect,1)
	end
