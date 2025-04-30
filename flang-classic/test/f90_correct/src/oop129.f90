! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

	program p
	USE CHECK_MOD
	type dt
        integer i(10)
        integer j(10)
        end type dt
        type(dt) d
        integer expect(3)
	integer results(3)
	do ii=1, 10
	associate (x => d%i(ii), y => d%j(ii))
          x = ii
          y = 10+ii
	end associate
        enddo
	expect = .true.
	associate (x => d%i, y => d%j)
	results(1) = all( x .eq. d%i)
	results(2) = all( y .eq. d%j)
	x = y
	results(3) = all(x .eq. d%j)
	end associate
        call check(results,expect,3)
	end program
