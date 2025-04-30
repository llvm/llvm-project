!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!* Attribute oriented initializations using experessions continaing intrinsic functions


program e11

 interface
  subroutine copy_str_to_result( str, result)
    integer :: result(:)
    character(len=*) :: str
  end subroutine
 end interface

  parameter(NTEST=68)
  integer :: result(NTEST)
  integer :: expect(NTEST) = (/  &
  !tinst%i1, tinst%i2
       1, 2,            &
  !iparm1
      -1,              &
  !iparm2
      65,              &
  !iparm3
      -65,             &
  !iparm4
      97,              &
  !iparm5
      -32,             &
  !iparm6
      -32,              &
  !c
      40,               &
  !i
      40,               &
  !ishft0
      40,               &
  !ishft1
      40,               &
  !ishft2
      -80,              &
  ! ishftarry0
      80, 80, 80, 80,           &
  ! ishftarry1
      40, 40, 40, 40,           &
  ! iparamarry
      1, 2, 3, 4,               &
  !ishft3
      4, 8, 12, 16,             &
  !iparam1
      40, 40, 40, 40,           &
  !iparam2
      -40,              &
  !iachar1
      88,               &
  !iachar1a
      88,               &
  !ishft4
      88,               &
  !iachar2
      -88,              &
  !iachararry1
      97, 98, 99, 48, 49,               &
  !iachararry2
      97, 98, 99, 48, 49,               &
  !iachararry3
      97, 98, 99, 48, 49,               &
  !iachararry4
      -97, -98, -99, -48, -49,          &
  !iachararry4a
      -97, -98, -99, -48, -49,          &
  !iachararry5
      -97, -98, -99, -48, -49          &
  /)

  type t
    integer::i1
    integer::i2
  end type
  type(t), parameter :: tinst = t(1,2)
  integer, parameter :: iparm1 = tinst%i1 - tinst%i2

  integer, parameter :: iparm2 = ichar('A')
  integer, parameter :: iparm3 = -ichar('A')
  integer, parameter :: iparm4 = ichar('a')
  integer, parameter :: iparm5 = iparm2  - iparm4
  integer, parameter :: iparm6 = ichar('A')-ichar('a')

  character, parameter :: c = char(40)
  integer, parameter :: i = ichar('(')
  integer, parameter :: ishft0 = ishft(5,3)
  integer, parameter :: ishft1 = ishft(5,ichar(char(3)))
  integer, parameter :: ishft2 = -ishft(5, 4) !ichar(char(3)))
  integer, parameter :: ishftarry0(4) = ishft(5,4)
  integer, parameter :: ishftarry1(4) = ishft(5,ichar(char(3)))
  integer, parameter :: iparamarry(4) = (/1,2,3,4/)
  integer, parameter :: ishft3(4) = ishft(iparamarry, 2)
  integer, parameter :: iparam1(4) = ichar(char(ishft1))
  integer, parameter :: iparam2 = -ichar(char(ishft1))

  character, parameter :: carray(5) = (/'a','b','c','0','1'/)
  integer, parameter :: iarray(5) = iachar((/'a','b','c','0','1'/))
  integer :: iachar1 = iachar('X')
  integer :: iachar1a = ichar('X')
  integer, parameter :: ishftparam1 = ishft(5,ichar(char(3)))
  integer :: ishft4 = ichar(char(88))
  integer :: iachar2 = -iachar('X')
  integer :: iachararry1(5) = iachar((/'a','b','c','0','1'/))
  integer :: iachararry2(5) = iachar((/carray/))
  integer :: iachararry3(5) = iachar(carray)
  integer :: iachararry4(5) = -iarray
  integer :: iachararry4a(5) = -iachar(carray)
  integer :: iachararry5(5) = -iachar((/'a','b','c','0','1'/))



!  print *,"! tinst%i1", tinst%i1
  result(1) = tinst%i1
!  print *,"! tinst%i2", tinst%i2
  result(2) = tinst%i2
!  print *,"! iparm1"
!  print *,iparm1;
  result(3) = iparm1
!  print *,"! iparm2"
!  print *,iparm2;
  result(4) = iparm2
!  print *,"! iparm3"
!  print *,iparm3;
  result(5) = iparm3
!  print *,"! iparm4"
!  print *,iparm4;
  result(6) = iparm4
!  print *,"! iparm5"
!  print *,iparm5;
  result(7) = iparm5
!  print *,"! iparm6"
!  print *,iparm6;
  result(8) = iparm6

!  print *,"! c"
!  print *,c;
  result(9) = iachar(c)
!  print *,"! i"
!  print *,i;
  result(10) = i
!  print *,"! ishft0"
!  print *,ishft0;
  result(11) = ishft0
!  print *,"! ishft1"
!  print *,ishft1;
  result(12) = ishft1
!  print *,"! ishft2"
!  print *,ishft2;
  result(13) = ishft2
!  print *,"!  ishftarry0"
!  print *, ishftarry0;
  result(14:17) =  ishftarry0
!  print *,"!  ishftarry1"
!  print *, ishftarry1;
  result(18:21) =  ishftarry1
!  print *,"!  iparamarry"
!  print *, iparamarry;
  result(22:25) =  iparamarry
!  print *,"! ishft3"
!  print *,ishft3;
  result(26:29) = ishft3
!  print *,"! iparam1"
!  print *,iparam1;
  result(30:33) = iparam1
!  print *,"! iparam2"
!  print *,iparam2;
  result(34) = iparam2

!  print *,"!iachar1"
!  print *,iachar1;
  result(35) = iachar1
!  print *,"!iachar1a"
!  print *,iachar1a;
  result(36) = iachar1a
!  print *,"!ishft4"
!  print *,ishft4;
  result(37) = ishft4
!  print *,"!iachar2"
!  print *,iachar2;
  result(38) = iachar2
!  print *,"!iachararry1"
!  print *,iachararry1;
  result(39:43) = iachararry1
!  print *,"!iachararry2"
!  print *,iachararry2;
  result(44:48) = iachararry2
!  print *,"!iachararry3"
!  print *,iachararry3;
  result(49:53) = iachararry3
!  print *,"!iachararry4"
!  print *,iachararry4;
  result(54:58) = iachararry4
!  print *,"!iachararry4a"
!  print *,iachararry4a;
  result(59:63) = iachararry4a
!  print *,"!iachararry5"
!  print *,iachararry5;
  result(64:68) = iachararry5

call check(result,expect,NTEST)

end program
