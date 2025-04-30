!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Tests scalar mask for sum intrinsic.


	program p

	integer stuff(3,3)
	logical(2) msk1(3,3),msk2(3,3)
	logical expect(2)
	logical rslt(2)
	data expect /.true.,.true./
	integer atrue(3), afalse(3)
	integer sfalse(3), strue(3)
	

	do i=1,3
	do j=1,3
	stuff(i,j) = i
	enddo
	enddo

	msk1 = .true.
	msk2 = .false.

	atrue = sum(stuff,dim=1,mask=msk1)
	strue = sum(stuff,dim=1,mask=.true.)
	afalse = sum(stuff,dim=1,mask=msk2)
	sfalse = sum(stuff,dim=1,mask=.false.)
	rslt(1) = all(atrue .eq. strue)
	rslt(2) = all(afalse .eq. sfalse)
	call check(rslt,expect,2)

	end

