!*** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!*** See https://llvm.org/LICENSE.txt for license information.
!*** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test pointer assignments with derived types and associated function
!

	integer,parameter::n=18
	integer result(n),expect(n)

	type dtype
	 integer,pointer :: am
	end type
	type etype
	 integer :: am
	end type
	type(dtype) :: d
	type(dtype),pointer :: dp
	type(etype),target :: et

	integer,pointer :: ap,bp
	integer,target :: at

	expect(:) = 1
	result(:) = 1

	at = 1

	ap => d%am
	if( associated(ap) ) then
	    result(1) = 0
	    print *,'ap should not be associated (1)'
	endif

	d%am => at

	if( .not. associated(d%am) ) then
	    result(2) = 0
	    print *,'d%am should be associated (2)'
	endif
	if( .not.associated(d%am,at) ) then
	    result(3) = 0
	    print *,'d%am should be associated with at (3)'
	endif
	ap => d%am
	if( .not.associated(ap) ) then
	    result(4) = 0
	    print *,'ap should be associated (4)'
	endif
	if( .not.associated(ap,d%am) ) then
	    result(5) = 0
	    print *,'ap should be associated with d%am (5)'
	endif
	if( .not.associated(ap,at) ) then
	    result(6) = 0
	    print *,'ap should be associated with at (6)'
	endif

	allocate(d%am)
	if( .not. associated(d%am) ) then
	    result(7) = 0
	    print *,'d%am should be associated (7)'
	endif
	ap => d%am
	if( .not.associated(ap) ) then
	    result(8) = 0
	    print *,'ap should be associated (8)'
	endif
	if( .not.associated(ap,d%am) ) then
	    result(9) = 0
	    print *,'ap should be associated with d%am (9)'
	endif

	allocate(dp)
	dp%am => at
	if( .not. associated(dp%am) ) then
	    result(10) = 0
	    print *,'dp%am should be associated (10)'
	endif
	if( .not.associated(dp%am,at) ) then
	    result(11) = 0
	    print *,'dp%am should be associated with at (11)'
	endif
	ap => dp%am
	if( .not.associated(ap) ) then
	    result(12) = 0
	    print *,'ap should be associated (12)'
	endif
	if( .not.associated(ap,dp%am) ) then
	    result(13) = 0
	    print *,'ap should be associated with dp%am (13)'
	endif
	if( .not.associated(ap,at) ) then
	    result(14) = 0
	    print *,'ap should be associated with at (14)'
	endif

	et%am = 99
	ap => et%am
	if( .not.associated(ap) ) then
	    result(15) = 0
	    print *,'ap should be associated (15)'
	endif
	if( .not.associated(ap,et%am) ) then
	    result(16) = 0
	    print *,'ap should be associated with et%am (16)'
	endif

	dp%am => et%am
	if( .not.associated(dp%am) ) then
	    result(17) = 0
	    print *,'dp%am should be associated (19)'
	endif
	if( .not.associated(dp%am,et%am) ) then
	    result(18) = 0
	    print *,'dp%am should be associated with et%am (18)'
	endif

	call check(result,expect,n)
        end
