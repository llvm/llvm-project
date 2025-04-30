!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!  'partisn' problem
!  use of a module with only clause for name that is both
!  a module variable and a member in the module
	module part
	integer :: ib           ! ib is a module variable
	type btype
	   integer ib           ! ib is also a member
	endtype
	type(btype)::bb
	end module

	subroutine sub1
	use part, only: ib       ! ib appears in the only clause
	implicit none
	ib = 99                 ! warning & we create a local ib
	end

	subroutine sub2
	use part, only: ib=>bb, jb=>ib       ! ib appears in the only clause
	ib%ib = jb
	end

	program p
	use part
	integer result(2),expect(2)
	data expect/99,99/
	bb%ib = 0
	ib = 0
	call sub1
	call sub2
!	print *,ib
!	print *,bb%ib
	result(1) = ib
	result(2) = bb%ib
	call check(result,expect,2)
	end
	

