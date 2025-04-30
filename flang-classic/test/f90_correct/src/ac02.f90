!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! correctly handle subroutines that appearly entirely in an include file

	include 'ac02.h'

	real e,r
	r = 0
	e = 1
	call s(r)
	call t(r)
	call check(r,e,1)
	end
