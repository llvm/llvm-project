!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Tests scalar mask for maxloc intrinsic.


	program p
	integer a(10,10)
	logical(8) msk(10,10)
	logical(8) rslt(2)
	logical(8) expect(2)
	integer afalse(10), sfalse(10)
	integer atrue(10), strue(10)
	data expect /.true.,.true./	

        do j=1,10	
	do i=1,10
	a(i,j) = i
	enddo
	enddo

	msk = .false.
	afalse = maxloc(a,dim=1,mask=msk)
	sfalse = maxloc(a,dim=1,mask=.false.)
	rslt(1) = all(afalse .eq. sfalse)
	strue = maxloc(a,dim=1,mask=.true.)
	msk = .true.
	atrue = maxloc(a,dim=1,mask=msk)
	rslt(2) = all(atrue .eq. strue)
	call check(rslt,expect,2)
	end
