! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test that format labels in contained subprograms do not conflict 
! with labels in the main prog

	character*5, s,t,u,v
	integer result(7)
	integer expect(7)
	data expect/1,1,1,1,1,1,1/
	result = 0
	write(s, 100)
	if( s .eq. 'glob' ) result(1) = 1
	call sub1
	write(t, 100)
	if( t .eq. 'glob' ) result(3) = 1
	call sub3
	write(u, 100)
	if( u .eq. 'glob' ) result(5) = 1
	call sub2
	write(v, 100)
	if( v .eq. 'glob' ) result(7) = 1
100	format('glob')
!	print *,result
	call check(result,expect,7)
	contains
	 subroutine sub1
	  character*5, s
	  write(s, 100)
	  if( s .eq. 'sub1' ) result(2) = 1
100	  format('sub1')
	 end subroutine
	 subroutine sub3
	  character*5, s
	  write(s,100)
	  if( s .eq. 'glob' ) result(4) = 1
	 end subroutine
	 subroutine sub2
	  character*5, s
	  write(s, 100)
100	  format('sub2')
	  if( s .eq. 'sub2' ) result(6) = 1
	 end subroutine
	end
