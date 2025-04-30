!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test that parenthesized arguments get copied to temps properly
!
        subroutine s1(i,r)
	integer i,r
        i = i + 1
        r = i
        end

        subroutine s2(i,j,r1,r2)
	integer i,j,r1,r2
        i = i + 1
        r1 = i
	r2 = j
        end

	integer,parameter::n=12
	integer result(n),expect(n)
	data expect/1,1,1,0,1,0,1,0,2,1,3,2/
	integer j
	! should pass zeroes both times
        call s1((0),result(1))
        call s1((0),result(2))

	call s2((0),0,result(3),result(4))
	call s2((0),0,result(5),result(6))

	j = 0

	call s2(j,(j),result(7),result(8))
	call s2(j,(j),result(9),result(10))
	call s2(j,(j),result(11),result(12))

	!print *,result
	call check(result,expect,n)
        end
