!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! From pwsfC:
! array-valued functions whose size depends on the size of a
! an argument whose correspoinding actual is a module allocatable
! array.
	module fn16
	real, allocatable :: zzz(:,:)
	contains
	    function ff1(xx)
	    real :: xx(:,:)
	    real :: ff1(size(xx))
	    ff1 = 11
	    endfunction
	    function ff2(xx)
	    real :: xx(:,:)
	    real :: ff2(size(xx,1))
	    ff2 = 13
	    endfunction
	    function ff3(xx,i)
	    real :: xx(:,:)
	    real :: ff3(size(xx,i))
	    ff3 = 17
	    endfunction
	endmodule
	subroutine sub(ii)
	use fn16
	integer :: result(3)
	integer :: expect(3)=(/66,26,51/)
	allocate(zzz(2,3))
	result(1) = sum(ff1(zzz))
	result(2) = sum(ff2(zzz))
	result(3) = sum(ff3(zzz,ii))
	call check(result, expect, 3)
	end
	call sub(2)
	end
