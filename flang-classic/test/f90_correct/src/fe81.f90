! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! tests kind argument for len,size,lbound,ubound,shape,maxloc,minloc (F2003)

	program p
	integer a(5,7)
        integer b(10)
	integer d(2:11,-1:81)
	integer bnd(2)
	integer bnd2(1)
	character*8 c
	integer result(39), expect(39)
	data expect /8,8,8,8,8,8,8,1,2,4,8,1,2,4,8,2,-1,2,-1,2,-1,2,-1,11,81,11,81,11,81,11,81,2,2,2,2,1,2,4,8/
		
	do i=1,10
	   b(i) = 10-i
	enddo
	result = 0
	
	result(1) = kind(len(c,kind=selected_int_kind(12)))
	result(2) = kind(size(a,1,kind=selected_int_kind(12)))
        result(3) = kind(lbound(array=a,dim=1,kind=selected_int_kind(12)))
	result(4) = kind(ubound(array=a,dim=1,kind=selected_int_kind(12)))
	result(5) = kind(shape(a,kind=selected_int_kind(12)))
	result(6) = kind(maxloc(array=b,kind=selected_int_kind(12)))
	result(7) = kind(minloc(array=b,kind=selected_int_kind(12)))

        result(8) = kind(lbound(array=d,kind=1))
        result(9) = kind(lbound(array=d,kind=2))
        result(10) = kind(lbound(array=d,kind=4))
        result(11) = kind(lbound(array=d,kind=8))

        result(12) = kind(ubound(array=d,kind=1))
        result(13) = kind(ubound(array=d,kind=2))
        result(14) = kind(ubound(array=d,kind=4))
        result(15) = kind(ubound(kind=8,array=d))

	bnd = 0
	bnd = lbound(array=d,kind=1)
	result(16) = bnd(1)
	result(17) = bnd(2)

	bnd = 0
	bnd = lbound(array=d,kind=2)
        result(18) = bnd(1)
        result(19) = bnd(2)

	bnd = 0
	bnd = lbound(array=d,kind=4)
        result(20) = bnd(1)
        result(21) = bnd(2)

	bnd = 0
	bnd = lbound(array=d,kind=8)
        result(22) = bnd(1)
        result(23) = bnd(2)

	bnd = 0
	bnd = ubound(array=d,kind=1)
        result(24) = bnd(1)
        result(25) = bnd(2)

	bnd = 0
        bnd = ubound(array=d,kind=2)
        result(26) = bnd(1)
        result(27) = bnd(2)

	bnd = 0
        bnd = ubound(array=d,kind=4)
        result(28) = bnd(1)
        result(29) = bnd(2)

	bnd = 0
        bnd = ubound(array=d,kind=8)
        result(30) = bnd(1)
        result(31) = bnd(2)
	
	bnd2 = 0
	bnd2 = shape(kind=1,source=bnd)
	result(32) = bnd2(1)
        bnd2 = 0
        bnd2 = shape(kind=2,source=bnd)
        result(33) = bnd2(1)
        bnd2 = 0
        bnd2 = shape(kind=4,source=bnd)
        result(34) = bnd2(1)
        bnd2 = 0
        bnd2 = shape(kind=8,source=bnd)
        result(35) = bnd2(1)

	result(36) = kind(shape(kind=1,source=bnd))
        result(37) = kind(shape(kind=2,source=bnd))
        result(38) = kind(shape(kind=4,source=bnd))
        result(39) = kind(shape(source=bnd,kind=8))

	call check(result,expect,39)
	
	end
