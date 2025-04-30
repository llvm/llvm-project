! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test that nested interfaces work as expected
program p
implicit none

 interface
  subroutine a(b,i)
  implicit none
  integer i
  interface
   integer function b(c)
   integer c
   end function b
  end interface
  end subroutine a
  integer function d(e)
   integer e
  end function
 end interface
 integer j
 integer result(1), expect(1)
 data expect/11/
 j = 1
 call a(d,j)
 result(1) = j
! print *,result,expect
 call check(result,expect,1)
end

subroutine a(b,i)
 implicit none
 integer i
 interface
  integer function b(c)
  integer c
  end function b
 end interface
 i = b(i)
end subroutine
integer function d(e)
 integer e
 d = 11*e
end function
