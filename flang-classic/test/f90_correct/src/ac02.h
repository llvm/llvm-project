!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! correctly handle subroutines that appearly entirely in an include file
! problem was ENLAB was only put out when nonzero STD_LINENO is seen,
! but STD_LINNO is not set for include file lines

	subroutine s(a)
	a = 1
	end

	subroutine t(a)
	if(a .gt. 0 ) goto 50
50	continue
	return
90	continue
	a = 2
!
! problem is caused by 'write', which sets STD_LINENO anyway,
! so ENLAB is put out for unreachable statement
!
	write(*,*)a
	return
100	continue
	return
	end
