!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Access host dummies from within the internal procedure.  The internal
! procedure needs to generate offsets from the 'display'.  The host
! dummies may be in stack of the host or the stack of the host's caller.
! The offsets can get messed up for 32-bit osx ABI (and newer 32-bit linux)
! where the stack is kept 16-byte aligned and the stack is 0mod16 at
! the point of the call instruction
!
	module zzz
	type myt
	    integer arr(100)
	    integer s1
	    integer s2
	endtype
	integer(4) res(6)
	contains
	    subroutine sub(rec, ii, cc)
		type(myt) :: rec
		integer ii
		character*(*) cc
		type(myt) :: lrg
		lrg%arr(99) = 787
		call internal
		contains
		subroutine internal
		    res(1) = rec%s1
		    res(2) = rec%s2
		    res(3) = ii
		    res(4) = len(cc)
		    res(5) = ichar(cc(1:1))
		    res(6) = ichar(cc(2:2))
		    end subroutine
	    end subroutine
	end module

	program test
	use zzz
	integer(4) exp(6)
	data exp/17, 23, 37, 2, 97, 98/
	type (myt) st
	st%s1 = 17
	st%s2 = 23
	call sub(st, 37, 'ab')
	call check(res, exp, 6)
!!	print 99, 'EXPECT: ', exp
!!	print 99, 'RESULT: ', res
!!99	format(a, 10i3)
	end
