! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

	program p
USE CHECK_MOD
	integer i(10)
	integer j(10)
        integer x
        integer expect(3)
	integer results(3)
	data i /1,2,3,4,5,6,7,8,9,10/
	data j /11, 12, 13, 14, 15, 16, 17, 18, 19, 20/
	x = 3
	expect = .true.
	assoc : associate (x => i, y => j)
	results(1) = all( x .eq. i)
	results(2) = all( y .eq. j)
	end associate assoc
	results(3) = x .eq. 3
        call check(results,expect,3)
	end program
