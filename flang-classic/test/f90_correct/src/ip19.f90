!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!	native pghpf problem:
!	host subprogram, both host and contained routines
!	need static-init routines
!
	subroutine ss(n,a)

	integer :: a(50)
	integer,save :: b(50)

!hpf$	distribute a(block),b(block)

	if( n .eq. 1 ) then
	 b = a
	 call sub( n )
	else
	 call sub( n )
	 a = b
	endif

	contains
	subroutine sub( n )
	integer,save :: c(50)
!hpf$	distribute c(block)

	if( n .eq. 1 ) then
	 c = a + b	! 1+1, 2+2, 3+3, ...
	else
	 b = b + c + a	! 1+2+1, 2+4+2, 3+6+3, ...
	endif
	end subroutine
	end

	subroutine tt(n,a)

	integer :: a(50)
	integer,save :: b(50)

!hpf$	distribute a(block),b(block)

	if( n .eq. 1 ) then
	 b = a
	 call sub( n )
	else
	 call sub( n )
	 a = b
	endif

	contains
	subroutine sub( n )
	integer,save :: c(50)
!hpf$	distribute c(block)

	if( n .eq. 1 ) then
	 c = a * b	! 1*1, 2*2, 3*3, ...
	else
	 b = b * c * a	! 1*1*1, 2*4*2, 3*9*3, ...
	endif
	end subroutine
	end


	program p
	integer a(50)
!hpf$	distribute a(block)
	integer result(2), expect(2)
	data expect/5100, 65666665/

	forall(i=1:50) a(i) = i

	call ss(1,a)
	call ss(2,a)
	!print *,a
	result(1) = sum(a)
	forall(i=1:50) a(i) = i
	call tt(1,a)
	call tt(2,a)
	!print *,a
	result(2) = sum(a)
	call check(result,expect,2)
	end
