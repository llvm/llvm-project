!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Tests scalar mask for minval intrinsic.


	program p
	integer a(10,10)
	logical(4) msk(10,10)
	integer afalse(10), sfalse(10)
        integer atrue(10), strue(10)
        logical(4) rslt(2)
        logical(4) expect(2)
        data expect /.true.,.true./

        do j=1,10	
	do i=1,10
	a(i,j) = i*2
	enddo
	enddo

	msk = .false.

!	do i=1,5
!	msk(i,1) = .true.
!	enddo
		
	sfalse = minval(a,dim=1,mask=.false.)
	afalse = minval(a,dim=1,mask=msk)
	strue = minval(a,dim=1,mask=.true.)
        msk = .true.
        atrue = minval(a,dim=1,mask=msk)
        rslt(1) = all(sfalse .eq. afalse)
        rslt(2) = all(atrue .eq. strue)
        call check(rslt,expect,2)
	end
