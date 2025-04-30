!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! check whether inlining with optional arguments works
! if the argument is not type scalar integer
! this test: inline with derived type arguments
module m
 type dt
  integer :: m1
  real :: m2
  integer, dimension(3) :: m3
 end type
 contains

 subroutine s1( a, b, c )
  type(dt) :: a
  integer :: b
  type(dt),optional :: c

  a%m1 = b+1
  a%m2 = b*2
  a%m3(1) = b+2
  a%m3(2) = b+3
  a%m3(3) = b+4
  if( present(c) ) then
   c%m1 = b+5
   c%m2 = b*3
   c%m3(1) = b+6
   c%m3(2) = b+7
   c%m3(3) = b+8
  endif
 end subroutine
end module

program p
 use m
 type(dt) a,c
 integer b
 integer result(15)
 integer expect(15)
 data expect /101,200,102,103,104,105, 300,106,107,108,201,400,202,203,204/

 b = 100
 call s1( a, b, c )
 result( 1) = a%m1
 result( 2) = a%m2
 result( 3) = a%m3(1)
 result( 4) = a%m3(2)
 result( 5) = a%m3(3)
 result( 6) = c%m1
 result( 7) = c%m2
 result( 8) = c%m3(1)
 result( 9) = c%m3(2)
 result(10) = c%m3(3)
 call s1( a, 200 )
 result(11) = a%m1
 result(12) = a%m2
 result(13) = a%m3(1)
 result(14) = a%m3(2)
 result(15) = a%m3(3)
 !print *,result
 call check(result,expect,15)
end program
