!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! check whether inlining with optional arguments works
! if the argument is not type scalar integer
! this test: inline with derived type array arguments
module m
 type dt
  integer :: m1
  real :: m2
  integer, dimension(3) :: m3
 end type
 contains

 subroutine s1( a, b, c )
  type(dt) :: a(3)
  integer :: b
  type(dt),optional :: c(2)

  a(1)%m1 = b+1
  a(1)%m2 = b*2
  a(1)%m3(1) = b+2
  a(1)%m3(2) = b+3
  a(1)%m3(3) = b+4
  a(2)%m1 = b+11
  a(2)%m2 = b*2 + 1
  a(2)%m3(1) = b+12
  a(2)%m3(2) = b+13
  a(2)%m3(3) = b+14
  a(3)%m1 = b+21
  a(3)%m2 = b*2 + 2
  a(3)%m3(1) = b+22
  a(3)%m3(2) = b+23
  a(3)%m3(3) = b+24
  if( present(c) ) then
   c(1)%m1 = b+5
   c(1)%m2 = b*3
   c(1)%m3(1) = b+6
   c(1)%m3(2) = b+7
   c(1)%m3(3) = b+8
   c(2)%m1 = b+15
   c(2)%m2 = b*3 + 8
   c(2)%m3(1) = b+16
   c(2)%m3(2) = b+17
   c(2)%m3(3) = b+18
  endif
 end subroutine
end module

program p
 use m
 type(dt) a(3),c(2)
 integer b
 integer result(40)
 integer expect(40)
 data expect /101,200,102,103,104,111,201,112,113,114,121,202, &
              122,123,124,105,300,106,107,108,115,308,116,117, &
              118,201,400,202,203,204,211,401,212,213,214,221, &
              402,222,223,224/


 b = 100
 call s1( a, b, c )
 result( 1) = a(1)%m1
 result( 2) = a(1)%m2
 result( 3) = a(1)%m3(1)
 result( 4) = a(1)%m3(2)
 result( 5) = a(1)%m3(3)
 result( 6) = a(2)%m1
 result( 7) = a(2)%m2
 result( 8) = a(2)%m3(1)
 result( 9) = a(2)%m3(2)
 result(10) = a(2)%m3(3)
 result(11) = a(3)%m1
 result(12) = a(3)%m2
 result(13) = a(3)%m3(1)
 result(14) = a(3)%m3(2)
 result(15) = a(3)%m3(3)
 result(16) = c(1)%m1
 result(17) = c(1)%m2
 result(18) = c(1)%m3(1)
 result(19) = c(1)%m3(2)
 result(20) = c(1)%m3(3)
 result(21) = c(2)%m1
 result(22) = c(2)%m2
 result(23) = c(2)%m3(1)
 result(24) = c(2)%m3(2)
 result(25) = c(2)%m3(3)
 call s1( a, 200 )
 result(26) = a(1)%m1
 result(27) = a(1)%m2
 result(28) = a(1)%m3(1)
 result(29) = a(1)%m3(2)
 result(30) = a(1)%m3(3)
 result(31) = a(2)%m1
 result(32) = a(2)%m2
 result(33) = a(2)%m3(1)
 result(34) = a(2)%m3(2)
 result(35) = a(2)%m3(3)
 result(36) = a(3)%m1
 result(37) = a(3)%m2
 result(38) = a(3)%m3(1)
 result(39) = a(3)%m3(2)
 result(40) = a(3)%m3(3)
 !print *,result
 call check(result,expect,40)
end program
