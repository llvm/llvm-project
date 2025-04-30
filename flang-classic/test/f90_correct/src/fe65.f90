!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!	array and scalar conformance (logical)

 subroutine x(nphi)
   integer            :: nphi
   integer, parameter :: maxphi=100
   double precision   :: rmax(maxphi)
   double precision  rmod
   logical :: within(maxphi)

   within(1:nphi) = rmod .lt. rmax(1:nphi)   !! LEGAL
 end subroutine x

	integer res, exp
	res = 1
	data exp/1/
	call check(res,exp,1)
	end
