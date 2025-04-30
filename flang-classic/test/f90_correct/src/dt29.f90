!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!   character members
!
module abcd
   integer :: result
   integer :: expect=0
   integer, parameter :: ntests=1
   contains 
      subroutine pqrs(a,b)
         character(len=*),intent(in) :: a
         character(len=*),dimension(:), intent(in) :: b
         integer :: i
	 result = 0
!         write(*,*)a,(b(i),i=1,SIZE(b))
	 if (a .ne. 'asdf')            result = 1
	 if (b(1) .ne. 'lfjsa')        result =  result + 2
	 if (b(2) .ne. 'alkdjfsadlfj') result =  result + 4
!	 print *, result
      end subroutine pqrs
end module abcd
program test
   use abcd
   type uvwx
      character(len=20), dimension(2) :: ijkl
   end type
   type(uvwx) :: qrst   
   qrst%ijkl(1)='lfjsa'
   qrst%ijkl(2)='alkdjfsadlfj'
   call pqrs('asdf',(/qrst%ijkl(1),qrst%ijkl(2)/))
   call check(result, expect, ntests)
end program
