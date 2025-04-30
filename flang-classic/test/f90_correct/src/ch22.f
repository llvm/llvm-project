! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! achar intrinsic, array-valued
!

	integer :: result(5)
	integer :: expect(5)
	character*5 aa
	call forall_bug(aa)
	do i = 1, 5
	    result(i) = iachar(aa(i:i))
	enddo

	data expect/97,98,99,100,101/
!	print *, result
!	print *, expect
	call check(result, expect, 5)
	end

	subroutine forall_bug(word2)
	  implicit none
	  
	  integer :: i, ibuf(5)
	  character(len=5) :: word1='abcde'
	  character(len=1) :: word2(5)


	  forall(i=1:5) ibuf(i)=iachar(word1(i:i))
	  word2 = achar(ibuf)

!!	  do i=1,5
!!	     ibuf(i)=iachar(word1(i:i))
!!	     word2(i:i)=achar(ibuf(i))
!!	  end do

!	  write(*,*) word1
!	  write(*,*) word2
	end
