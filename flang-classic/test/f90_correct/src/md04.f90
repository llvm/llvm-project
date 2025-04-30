! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test that contained subprograms and module contained subprograms
! and contained subprograms in module contained subprograms are resolved 
! properly
!
! this came from an unrelated bug report from Intel

module modB
 integer,parameter::ntests=11
 logical,parameter::debug=.false.
 integer result(ntests), n
contains
 subroutine register(r)
  integer r
  n = n + 1
  if( n .le. ubound(result,1) ) then
   result(n) = r
  else
   result(ubound(result,1)) = -1
  endif
 end subroutine
end module modB

module modA
 use modB
contains
 subroutine sub1
  if( debug ) print *,'in moda.sub1'
  call register(2)
  call sub2   ! calls subroutine sub2 internal to this subroutine

 contains
  subroutine sub2
   if( debug ) print *,'in moda.sub1.sub2'
   call register(3)
  end subroutine sub2
 end subroutine sub1
end module modA

program main
 use modA
 use modB
 integer expect(ntests)
 data expect/1,2,3,6,4,5,7,8,2,3,0/

 if( debug ) print *,'in main program'
 call register(1)
 call sub1    ! calls sub1 in modA
 call sub2    ! calls external subroutine sub2
 call sub3    ! calls external sub3
 if( debug ) print *,'finished in main program'
 call register(0)
! print *,result
 call check(result,expect,ntests)
end program main


subroutine sub1
 use modB
 if( debug ) print *,'in sub1'
 call register(4)
 call sub2   ! calls subroutine sub2 internal to this subroutine

contains
 subroutine sub2
  if( debug ) print *,'in sub1.sub2'
  call register(5)
 end subroutine sub2
end subroutine sub1

subroutine sub2
 use modB
 if( debug ) print *,'in sub2'
 call register(6)
 call sub1      ! calls external subroutine sub1
end subroutine sub2

subroutine sub3
 use modB
 use modA
 if( debug ) print *,'in sub3'
 call register(7)
 call sub2
 call sub1
contains
 subroutine sub2
  if( debug ) print *,'in sub3.sub2'
  call register(8)
 end subroutine
end subroutine
