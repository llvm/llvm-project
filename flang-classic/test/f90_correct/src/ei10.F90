! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Attribute oriented initializations using intrinsic functions

program e10

 interface
  subroutine copy_str_to_result( str, result)
    integer :: result(:)
    character(len=*) :: str
  end subroutine
 end interface

   parameter(NTEST=418)
   integer :: result(NTEST)
   integer :: expect(NTEST) = (/  &
   !int1
       3,		&
   !int2
       42,		&
   !int3
       1,		&
   !intarry1
       1, 2,		&
   !intarry2
       3, 4,		&
   !nint1
       3,		&
   !nint2
       43,		&
   !nintarry1
       1, 3,		&
   !nintarry2
       4, 4,		&
   !uboundarry1
       5, 2,		&
   !uboundarry2
       5, 2,		&
   !lboundarry1
       1, 1,		&
   !lboundarry2
       1, 1,		&
   !adjustl1
       97, 98, 99, 32, 32, 32,            &
   !adjustl2
       97, 98, 99, 32, 32, 32,		&
   !adjustl3
       97, 98, 99, 32, 32, 32,		&
   !adjustl4
       97, 98, 99, 32, 32, 32,		&
   !adjustr1
       32, 32, 32, 97, 98, 99,		&
   !adjustr2
       32, 32, 32, 97, 98, 99,		&
   !adjustr3
       32, 32, 32, 97, 98, 99,		&
   !adjustr4
       32, 32, 32, 97, 98, 99,		&
   !t1_param2%str1
       97, 98, 99, 32, 32, 32,		&
   !t1_param3%str1
       32, 32, 32, 97, 98, 99,		&
   !t1_param4%str1
       32, 32, 32, 97, 98, 99,		&
   !adjustlarry1(1)
       97, 98, 99, 122, 32, 32,		&
   !adjustlarry1(2)
       97, 98, 99, 32, 32, 32,		&
   !adjustlarry1(3)
       97, 98, 99, 32, 32, 32,		&
   !adjustlarry1(4)
       97, 98, 99, 32, 32, 32,		&
   !adjustrarry1(1)
       32, 32, 97, 98, 99, 122,		&
   !adjustrarry1(2)
       32, 32, 32, 97, 98, 99,		&
   !adjustrarry1(3)
       32, 32, 32, 97, 98, 99,		&
   !adjustrarry1(4)
       32, 32, 32, 97, 98, 99,		&
   !idx1
       5,		&
   !idx2
       5,		&
   !idx3
       6,		&
   !idx4
       6,		&
   !idx5
       5,		&
   !idx6
       5,		&
   !idx7
       5,		&
   !idx8
       5,		&
   !idx9
       2,		&
   !idxarry1
       2, 3, 4, 5,		&
   !idxarry2
       2, 3,		&
   !lentrim1
       6,		&
   !lentrim2
       13,		&
   !lentrim3
       6,		&
   !lentrim4
       3,		&
   !lentrimarry1
       4, 4, 5, 6,		&
   !lentrimarry2
       3, 4,		&
   !repeatstr1
       32, 97, 98, 99, 32, 32, 32, 97, &
       98, 99, 32, 32, 32, 97, 98, 99, &
       32, 32, &
   !repeatstr2
       32, 32, 32, 97, 98, 99, 32, 32, &
       32, 97, 98, 99, 32, 32, 32, 97, &
       98, 99,  &
   !scan1
       5,		&
   !scan2
       6,		&
   !scan3
       6,		&
   !scan4
       6,		&
   !scan5
       5,		&
   !scan6
       6,		&
   !scan7
       6,		&
   !scan8
       5,		&
   !scan9
       2,		&
   !scanarry1
       2, 3, 4, 5,		&
   !scanarry2
       2, 4,		&
   !vrfy1
       0,		&
   !vrfy2
       10,		&
   !vrfy3
       1,		&
   !vrfy4
       1,		&
   !vrfy5
       1,		&
   !vrfy6
       3,		&
   !vrfy7
       4,		&
   !vrfy8
       1,		&
   !vrfy9
       1,		&
   !vrfyarry1
       4, 0, 0, 0,		&
   !vrfyarry2
       4, 1,		&
   !trim1
       120, 120, 121, 121, 98, 99,		&
   !trim2
       122, 120, 120, 121, 121, 98, 99,	&
       119, 119, 118, 118, 122, 122,		&
   !trim3
       32, 32, 32, 97, 98, 99,		&
   !trim4
       32, 32, 32, 97, 98, 99, 32,   &
       32, 32, 32, 32,  &
   !char1
       65,		&
   !char2
       66,		&
   !chararry1
       48, 49, 50, 51, 52, 53,		&
   !ichar1
       88,		&
   !ichar2
       90,		&
   !ichararry1
       97, 98, 99, 48, 49,		&
   !iachar1
       88,		&
   !iachararry1
       97, 98, 99, 48, 49,		&
   !select_ikind1
       1,		&
   !select_ikind2
       2,		&
   !select_ikind3
       2,		&
   !select_ikind4
       4,		&
   !select_ikind5
       4,		&
   !select_ikind6
       8,		&
   !select_ikind7
       8,		&
   !select_ikind8
       -1,		&
   !select_ikind9
       4,		&
   !select_rkind1
       4,		&
   !select_rkind2
       4,		&
   !select_rkind3
       8,		&
   !select_rkind4
       8,		&
#ifdef __flang_quadfp__
   !select_rkind5
       16,		&
#else
   !select_rkind5
       -1,		&
#endif
   !select_rkind6
       4,		&
   !select_rkind7
       4,		&
   !select_rkind8
       4,		&
   !select_rkind9
       8,		&
   !select_rkind10
       8,		&
#ifdef __flang_quadfp__
   !select_rkind11
       16,		&
#else
   !select_rkind11
       -2,		&
#endif
   !select_rkind12
       -2,		&
   !select_rkind13
       4,		&
   !select_rkind14
       4,		&
   !select_rkind15
       4,		&
   !select_rkind16
       4,		&
   !select_rkind17
       8,		&
   !select_rkind18
       8,		&
#ifdef __flang_quadfp__
   !select_rkind19
       16,		&
   !select_rkind20
       16,		&
#else
   !select_rkind19
       -2,		&
   !select_rkind20
       -3,		&
#endif
   !select_rkind21
       -2,		&
#ifdef __flang_quadfp__
   !select_rkind22
       -2,		&
#else
   !select_rkind22
       -3,		&
#endif
   !i2dimarryparam1
       50, 40, 30, 20, 10, 1,		&
       2, 3, 4, 5,		&
   !i2dimarryparam2
       50, 40, 30, 20, 10, 2,		&
       3, 4, 5, 6,		&
   !i2dimarryparam3
       50, 40, 30, 20, 10, 2,		&
       3, 4, 5, 6,		&
   !i2dimarryparam3
       50, 40, 30, 20, 10, 2,		&
       3, 4, 5, 6,		&
   !i2dimarryparam4
       50, 40, 30, 20, 10, 2,		&
       3, 4, 5, 6,		&
   !i2dimarryparam5
       50, 30, 10, 3, 5, 40,		&
       20, 2, 4, 6,		&
   !i2dimarryparam6
       50, 30, 10, 3, 5, 40,		&
       20, 2, 4, 6,		&
   !len1
       6,		&
   !len2
       2,		&
   !len3
       6,		&
   !len4
       6,		&
   !size2
       10,		&
   !size3
       384,		&
   !size4
       2,		&
   !size5
       4,		&
   !size6
       6,		&
   !size7
       8,		&
   !size8
       8,		&
   !size9
       2,		&
   !ishft1
       20,		&
   !ishft2
       64,		&
   !ishftarry1
       4, 8, 12, 16, 20,		&
   !ishftarry2
       24, 24, 25, 26, 26,		&
   !ishftarry3
       16, 24, 32, 40, 48,		&
   !ishftarry4
       96, 98, 100, 104, 106		&
   /)

   integer, parameter :: iparam = 8
   integer, parameter :: iparamarry(5) = (/(i,i=1,5)/)
   character(len=6), parameter :: strparamarry1(4) = &
                    (/"abcz  ", " abc  ", "  abc ", "   abc" /)
   character(len=6), parameter :: strparam1 = "   abc"
   character(len=2), parameter :: strparam2 = "bc"
   character(len=6) :: adjustl1 = adjustl("   abc")
   character(len=6) :: adjustl2 = adjustl(strparam1)
   character(len=6) :: adjustlarry1(4)  = adjustl(strparamarry1)
   character(len=6) :: adjustrarry1(4)  = adjustr(strparamarry1)

   character(len=6), parameter :: strparam3 = "   abc"
   character(len=6) :: adjustr1 = adjustr("abc   ")
   character(len=6) :: adjustr2 = adjustr(strparam3)

  type t1
    character(len=6) :: str1
  end type
  type (t1) :: t1_inst1
  type (t1), parameter:: t1_param1 = t1( adjustl("   abc") )
  type (t1), parameter:: t1_param2 = t1( adjustl(strparam1) )
  type (t1), parameter:: t1_param5 = t1( 'abcdef' )
  character(len=6) :: adjustl3 = adjustl(t1_param1%str1)
  character(len=6) :: adjustl4 = adjustl(t1_param2%str1)
  type (t1), parameter:: t1_param3 = t1( adjustr("abc   ") )
  type (t1), parameter:: t1_param4 = t1( adjustr(strparam3) )

  type t2
    integer :: t2_i
  end type
  type (t2), parameter :: t2_param =  t2(9)

  character(len=6) :: adjustr3 = adjustr(t1_param3%str1)
  character(len=6) :: adjustr4 = adjustr(t1_param4%str1)
  integer :: idx1 = index("xxyybcwwvv", "bc")
  integer :: idx2 = index("xxyybcwwvv", "bc", .TRUE.)
  integer :: idx3 = index("zxxyybcwwvvzz", "bc", .FALSE.)
  integer :: idx4 = index("zxxyybcwwvvzz", strparam2, .FALSE.)
  integer :: idx5 = index(strparam1, "bc")
  integer :: idx6 = index(strparam1, "bc", .TRUE.)
  integer :: idx7 = index(strparam1, strparam2, .TRUE.)
  integer :: idx8 = index(strparam1, "bc", .FALSE.)
  integer :: idx9 = index(t1_param1%str1, "bc", .FALSE.)
  integer :: idxarry1(4) = index(strparamarry1, "bc")
  integer :: idxarry2(2) = index((/"abc   ", " abc  "/), "bc")

  integer :: scan1 = scan("xxyybcwwvv", "bc")
  integer :: scan2 = scan("xxyybcwwvv", "bc", .TRUE.)
  integer :: scan3 = scan("zxxyybcwwvvzz", "bc", .FALSE.)
  integer :: scan4 = scan("zxxyybcwwvvzz", strparam2, .FALSE.)
  integer :: scan5 = scan(strparam1, "bc")
  integer :: scan6 = scan(strparam1, "bc", .TRUE.)
  integer :: scan7 = scan(strparam1, strparam2, .TRUE.)
  integer :: scan8 = scan(strparam1, "bc", .FALSE.)
  integer :: scan9 = scan(t1_param1%str1, "bc", .FALSE.)
  integer :: scanarry1(4) = scan(strparamarry1, "bc")
  integer :: scanarry2(2) = scan((/"abc   ", " aac  "/), "bc")

  integer :: vrfy1 = verify("bbbbc", "bc")
  integer :: vrfy2 = verify("xxyybcwwvv", "xywzbc", .TRUE.)
  integer :: vrfy3 = verify("zxxyybcwwvvzz", "bc", .FALSE.)
  integer :: vrfy4 = verify("zxxyybcwwvvzz", strparam2, .FALSE.)
  integer :: vrfy5 = verify(strparam1, "abc")
  integer :: vrfy6 = verify(strparam1, "bac", .TRUE.)
  integer :: vrfy7 = verify(strparam1, strparam2, .TRUE.)
  integer :: vrfy8 = verify(strparam1, "abc", .FALSE.)
  integer :: vrfy9 = verify(t1_param1%str1, "bc", .FALSE.)
  integer :: vrfyarry1(4) = verify(strparamarry1, " abc")
  integer :: vrfyarry2(2) = verify((/"abc   ", " abc  "/), "abc")

  integer :: lentrim1 = len_trim("xxyybc    ")
  integer :: lentrim2 = len_trim("zxxyybcwwvvzz")
  integer :: lentrim3 = len_trim(strparam3)
  integer :: lentrim4 = len_trim(t1_param1%str1)
  integer :: lentrimarry1(4) = len_trim(strparamarry1)
  integer :: lentrimarry2(2) = len_trim((/"abc   ", " abc  "/))

  integer :: len1 = len(strparam1)
  integer :: len2 = len(strparam2)
  integer :: len3 = len(t1_param5%str1)
  integer :: len4 = len(t1_inst1%str1)
  integer :: len5 = len(strparamarry1(2))

  real, parameter :: fltparam1 = 42.52
  real, parameter :: fltparamarry1(2) = (/ 3.53, 4.44 /)
  complex, parameter :: cmplx1(2) = ( 3.53, 4.44 )

  integer :: int1 =  int(3.1417)
  integer :: int2 =  int(fltparam1)
  integer :: int3 =  int( (1.111, 2.222) )
  integer :: intarry1(2) =  int( (/1.111, 2.522/) )
  integer :: intarry2(2) =  int( fltparamarry1 )

  integer :: nint1 =  nint(3.1417)
  integer :: nint2 =  nint(fltparam1)
  integer :: nintarry1(2) =  nint( (/1.111, 2.522/) )
  integer :: nintarry2(2) =  nint( fltparamarry1 )

  integer :: i4dimarry3(2,4,6,8)
  integer :: size2 = size( reshape((/50,40,30,20,10,1,2,3,4,5/), (/5,2/)) );
  integer :: size3 = size(i4dimarry3);
  integer :: size4 = size(i4dimarry3,1);
  integer :: size5 = size(i4dimarry3,2);
  integer :: size6 = size(i4dimarry3,3);
  integer :: size7 = size(i4dimarry3,4);
  integer :: size8 = size(i4dimarry3,4);
  integer :: size9 = size( reshape((/50,40,30,20,10,1,2,3,4,5/), (/5,2/)),2 );

  character(len=6) :: trim1 = trim("xxyybc    ")
  character(len=13) :: trim2 = trim("zxxyybcwwvvzz")
  character(len=6) :: trim3 = trim(strparam3)
  character(len = 12) :: trim4 = trim(strparam3)
  character(len=3) :: trim5 = trim(t1_param1%str1)

  character(len=18) :: repeatstr1 = repeat(" abc  ", 3)
  character(len=18) :: repeatstr2 = repeat(strparam1, 3)

  integer, parameter :: iparam2 = 66
  character :: char1 = char(65)
  character :: char2 = char(iparam2)
  character :: chararry1(6) = char((/48,49,50,51,52,53/))

  character, parameter :: cparam ='Z'
  integer :: ichar1 = ichar('X')
  integer :: ichar2 = ichar(cparam)
  integer :: ichararry1(5) = ichar((/'a','b','c','0','1'/))		!ERR

  integer :: iachar1 = iachar('X')
  integer :: iachararry1(5) = iachar((/'a','b','c','0','1'/))

    integer :: select_ikind1 = selected_int_kind(2);
    integer :: select_ikind2 =  selected_int_kind(3);
    integer :: select_ikind3 =  selected_int_kind(4);
    integer :: select_ikind4 =  selected_int_kind(8);
    integer :: select_ikind5 =  selected_int_kind(9);
    integer :: select_ikind6 =  selected_int_kind(10);
    integer :: select_ikind7 =  selected_int_kind(18);
    integer :: select_ikind8 =  selected_int_kind(19);
    integer :: select_ikind9 =  selected_int_kind(t2_param%t2_i);!OK

    integer :: select_rkind1 =selected_real_kind(4);
    integer :: select_rkind2 = selected_real_kind(6);
    integer :: select_rkind3 = selected_real_kind(14);
    integer :: select_rkind4 = selected_real_kind(15);
    integer :: select_rkind5 = selected_real_kind(16);
    integer :: select_rkind6 = selected_real_kind(r=2);
    integer :: select_rkind7 = selected_real_kind(r=36);
    integer :: select_rkind8 = selected_real_kind(r=37);
    integer :: select_rkind9 = selected_real_kind(r=306);
    integer :: select_rkind10 = selected_real_kind(r=307);
    integer :: select_rkind11 = selected_real_kind(r=4931);
    integer :: select_rkind12 = selected_real_kind(r=4932);

    integer :: select_rkind13 = selected_real_kind(4,36);
    integer :: select_rkind14 = selected_real_kind(5,36);
    integer :: select_rkind15 = selected_real_kind(4,37);
    integer :: select_rkind16 = selected_real_kind(5,37);
    integer :: select_rkind17 = selected_real_kind(8,306);
    integer :: select_rkind18 = selected_real_kind(9,306);
    integer :: select_rkind19 = selected_real_kind(15,4931);
    integer :: select_rkind20 = selected_real_kind(16,4931);
    integer :: select_rkind21 = selected_real_kind(15,4932);
    integer :: select_rkind22 = selected_real_kind(18,4932);


   type t3
     integer :: t_iarry(5,2)
   end type
   type (t3), parameter:: t3_param1 = &		!ERR
             t3( reshape((/50,40,30,20,10,1,2,3,4,5/), (/5,2/)) )	!ERR

   type t4
     integer :: t_iarry(5)
   end type

   integer, parameter :: padparam(2) = (/1,-1/)
   integer, parameter :: shapeparam(2) = (/5,2/)
   integer, parameter :: orderparam(2) = (/2,1/)
   type (t4), parameter:: t4_param5 = t4( (/2,3,4,5,6/) )
   type (t4), parameter:: t_param = t4( (/1,2,3,4,5/) )
   integer, parameter :: i2dimarryparam1(5,2) = &
           reshape((/50,40,30,20,10,1,2,3,4,5/), (/5,2/))
   integer, parameter :: i2dimarryparam2(5,2)  =  &
            reshape((/50,40,30,20,10,t4_param5%t_iarry/), (/5,2/))
   integer, parameter :: i2dimarryparam3(5,2)  =  &
            reshape((/50,40,30,20,10,t4_param5%t_iarry/), shapeparam)

   type t5
     integer :: t_iarry1(2)
     integer :: t_iarry2(2)
   end type

   type (t5), parameter :: t5_param = t5((/5,2/), (/2,1/))
   integer, parameter :: i2dimarryparam4(5,2)  =  &
            reshape((/50,40,30,20,10,t4_param5%t_iarry/), t5_param%t_iarry1)
   integer, parameter :: i2dimarryparam5(5,2)  =  &
            reshape((/50,40,30,20,10,t4_param5%t_iarry/), &
            t5_param%t_iarry1, order = t5_param%t_iarry2)

   integer, parameter :: i2dimarryparam6(5,2)  =  &
            reshape((/50,40,30,20,10,t4_param5%t_iarry/), &
            t5_param%t_iarry1, order = (/2,1/))

   integer :: iidimarry(2)

  integer :: ishft1 = ishft(5,2);
  integer :: ishft2 = ishft(iparam,3);
  integer :: ishftarry1(5) =  ishft(iparamarry,2);		!ERR
  integer :: ishftarry2(5) =  ishft((/48,49,50,52,53/),-1);		!ERR
  integer :: ishftarry3(5) =  ishft(t4_param5%t_iarry,3);		!ERR
  integer :: ishftarry4(5) =  ishft((/48,49,50,52,53/),1);		!ERR

  integer :: uboundarry1(2)  = ubound(i2dimarryparam1)
  integer :: uboundarry2(2)  = ubound(i2dimarryparam2)

  integer :: lboundarry1(2)  = lbound(i2dimarryparam1)
  integer :: lboundarry2(2)  = lbound(i2dimarryparam2)

  integer, pointer :: iptr => NULL()		!ERR

!  print *,"! int1"
!  print *,int1;
  result(1) = int1
!  print *,"! int2"
!  print *,int2;
  result(2) = int2
!  print *,"! int3"
!  print *,int3;
  result(3) = int3
!  print *,"! intarry1"
!  print *,intarry1;
  result(4:5) = intarry1
!  print *,"! intarry2"
!  print *,intarry2;
  result(6:7) = intarry2

!  print *,"! nint1"
!  print *,nint1;
  result(8) = nint1
!  print *,"! nint2"
!  print *,nint2;
  result(9) = nint2
!  print *,"! nintarry1"
!  print *,nintarry1;
  result(10:11) = nintarry1
!  print *,"! nintarry2"
!  print *,nintarry2;
  result(12:13) = nintarry2

!  print *,"! uboundarry1"
!  print *,uboundarry1;
  result(14:15) = uboundarry1
!  print *,"! uboundarry2"
!  print *,uboundarry2;
  result(16:17) = uboundarry2

!  print *,"! lboundarry1"
!  print *,lboundarry1;
  result(18:19) = lboundarry1
!  print *,"! lboundarry2"
!  print *,lboundarry2;
  result(20:21) = lboundarry2

!  print *,"! adjustl1"
!  print *,adjustl1;
  call copy_str_to_result(adjustl1,result(22:27))
!  print *,"! adjustl2"
!  print *,adjustl2;
  call copy_str_to_result(adjustl2,result(28:33))
!  print *,"! adjustl3"
!  print *,"'", adjustl3, "'"
  call copy_str_to_result(adjustl3,result(34:39))
!  print *,"! adjustl4"
!  print *,"'", adjustl4, "'"
  call copy_str_to_result(adjustl4,result(40:45))

! PROBLEM in adjustr ???
!  print *,"! adjustr1"
!  print *,"'", adjustr1, "'"
  call copy_str_to_result(adjustr1,result(46:51))
!  print *,"! adjustr2"
!  print *,adjustr2;
  call copy_str_to_result(adjustr2,result(52:57))
!  print *,"! adjustr3"
!  print *,adjustr3;
  call copy_str_to_result(adjustr3,result(58:63))
!  print *,"! adjustr4"
!  print *,adjustr4;
  call copy_str_to_result(adjustr4,result(64:69))

!  print *,t1_param2%str1;
  call copy_str_to_result(t1_param2%str1,result(70:75))
!  print *,"! t1_param3%str1"
!  print *,t1_param3%str1;
  call copy_str_to_result(t1_param3%str1,result(76:81))
!  print *,"! t1_param4%str1"
!  print *,t1_param4%str1;
  call copy_str_to_result(t1_param4%str1,result(82:87))
!  print *,"! adjustlarry1(1)"
!  print *,adjustlarry1(1);
  call copy_str_to_result(adjustlarry1(1),result(88:93))
!  print *,"! adjustlarry1(2)"
!  print *,adjustlarry1(2);
  call copy_str_to_result(adjustlarry1(2),result(94:99))
!  print *,"! adjustlarry1(3)"
!  print *,adjustlarry1(3);
  call copy_str_to_result(adjustlarry1(3),result(100:105))
!  print *,"! adjustlarry1(4)"
!  print *,adjustlarry1(4);
  call copy_str_to_result(adjustlarry1(4),result(106:111))
!  print *,"! adjustrarry1(1)"
!  print *,adjustrarry1(1);
  call copy_str_to_result(adjustrarry1(1),result(112:117))
!  print *,"! adjustrarry1(2)"
!  print *,adjustrarry1(2);
  call copy_str_to_result(adjustrarry1(2),result(118:123))
!  print *,"! adjustrarry1(3)"
!  print *,adjustrarry1(3);
  call copy_str_to_result(adjustrarry1(3),result(124:129))
!  print *,"! adjustrarry1(4)"
!  print *,adjustrarry1(4);
  call copy_str_to_result(adjustrarry1(4),result(130:135))

!  print *,"! idx1"
!  print *,idx1;
  result(136) = idx1
!  print *,"! idx2"
!  print *,idx2;
  result(137) = idx2
!  print *,"! idx3"
!  print *,idx3;
  result(138) = idx3
!  print *,"! idx4"
!  print *,idx4;
  result(139) = idx4
!  print *,"! idx5"
!  print *,idx5;
  result(140) = idx5
!  print *,"! idx6"
!  print *,idx6;
  result(141) = idx6
!  print *,"! idx7"
!  print *,idx7;
  result(142) = idx7
!  print *,"! idx8"
!  print *,idx8;
  result(143) = idx8
!  print *,"! idx9"
!  print *,idx9;
  result(144) = idx9
!  print *,"! idxarry1"
!  print *,idxarry1;
  result(145:148) = idxarry1
!  print *,"! idxarry2"
!  print *,idxarry2;
  result(149:150) = idxarry2
!  print *,"! lentrim1"
!  print *,lentrim1;
  result(151) = lentrim1
!  print *,"! lentrim2"
!  print *,lentrim2;
  result(152) = lentrim2
!  print *,"! lentrim3"
!  print *,lentrim3;
  result(153) = lentrim3
!  print *,"! lentrim4"
!  print *,lentrim4;
  result(154) = lentrim4
!  print *,"! lentrimarry1"
!  print *,lentrimarry1;
  result(155:158) = lentrimarry1
!  print *,"! lentrimarry2"
!  print *,lentrimarry2;
  result(159:160) = lentrimarry2
!  print *,"! repeatstr1"
!  print *,repeatstr1;
  call copy_str_to_result(repeatstr1,result(161:178))
!  print *,"! repeatstr2"
!  print *,repeatstr2;
  call copy_str_to_result(repeatstr2,result(179:196))

!  print *,"! scan1"
!  print *,scan1;
  result(197) = scan1
!  print *,"! scan2"
!  print *,scan2;
  result(198) = scan2
!  print *,"! scan3"
!  print *,scan3;
  result(199) = scan3
!  print *,"! scan4"
!  print *,scan4;
  result(200) = scan4
!  print *,"! scan5"
!  print *,scan5;
  result(201) = scan5
!  print *,"! scan6"
!  print *,scan6;
  result(202) = scan6
!  print *,"! scan7"
!  print *,scan7;
  result(203) = scan7
!  print *,"! scan8"
!  print *,scan8;
  result(204) = scan8
!  print *,"! scan9"
!  print *,scan9;
  result(205) = scan9
!  print *,"! scanarry1"
!  print *,scanarry1;
  result(206:209) = scanarry1
!  print *,"! scanarry2"
!  print *,scanarry2;
  result(210:211) = scanarry2

!  print *,"! vrfy1"
!  print *,vrfy1;
  result(212) = vrfy1
!  print *,"! vrfy2"
!  print *,vrfy2;
  result(213) = vrfy2
!  print *,"! vrfy3"
!  print *,vrfy3;
  result(214) = vrfy3
!  print *,"! vrfy4"
!  print *,vrfy4;
  result(215) = vrfy4
!  print *,"! vrfy5"
!  print *,vrfy5;
  result(216) = vrfy5
!  print *,"! vrfy6"
!  print *,vrfy6;
  result(217) = vrfy6
!  print *,"! vrfy7"
!  print *,vrfy7;
  result(218) = vrfy7
!  print *,"! vrfy8"
!  print *,vrfy8;
  result(219) = vrfy8
!  print *,"! vrfy9"
!  print *,vrfy9;
  result(220) = vrfy9
!  print *,"! vrfyarry1"
!  print *,vrfyarry1;
  result(221:224) = vrfyarry1
!  print *,"! vrfyarry2"
!  print *,vrfyarry2;
  result(225:226) = vrfyarry2

!  print *,"! trim1"
!  print *,trim1;
  call copy_str_to_result(trim1, result(227:332))
!  print *,"! trim2"
!  print *,trim2;
  call copy_str_to_result(trim2, result(233:245))
!  print *,"! trim3"
!  print *,trim3;
  call copy_str_to_result(trim3, result(246:251))
!  print *,"! trim4"
!  print *,trim4;
  call copy_str_to_result(trim4, result(252:262))

!  print *,"! char1"
!  print *,char1;
  result(263) = iachar(char1)
!  print *,"! char2"
!  print *,char2;
  result(264) = iachar(char2)
!  print *,"! chararry1"
!  print *,chararry1;
  result(265:270) = iachar(chararry1)

!  print *,"! ichar1"
!  print *,ichar1;
  result(271) = ichar1
!  print *,"! ichar2"
!  print *,ichar2;
  result(272) = ichar2
!  print *,"! ichararry1"
!  print *,ichararry1;
  result(273:277) = ichararry1


!  print *,"! iachar1"
!  print *,iachar1;
  result(278) = iachar1
!  print *,"! iachararry1"
!  print *,iachararry1;
  result(279:283) = iachararry1

!  print *,"! select_ikind1"
!  print *,select_ikind1;
  result(284) = select_ikind1
!  print *,"! select_ikind2"
!  print *,select_ikind2;
  result(285) = select_ikind2
!  print *,"! select_ikind3"
!  print *,select_ikind3;
  result(286) = select_ikind3
!  print *,"! select_ikind4"
!  print *,select_ikind4;
  result(287) = select_ikind4
!  print *,"! select_ikind5"
!  print *,select_ikind5;
  result(288) = select_ikind5
!  print *,"! select_ikind6"
!  print *,select_ikind6;
  result(289) = select_ikind6
!  print *,"! select_ikind7"
!  print *,select_ikind7;
  result(290) = select_ikind7
!  print *,"! select_ikind8"
!  print *,select_ikind8;
  result(291) = select_ikind8
!  print *,"! select_ikind9"
!  print *,select_ikind9;
  result(292) = select_ikind9

!  print *,"! select_rkind1"
!  print *,select_rkind1;
  result(293) = select_rkind1
!  print *,"! select_rkind2"
!  print *,select_rkind2;
  result(294) = select_rkind2
!  print *,"! select_rkind3"
!  print *,select_rkind3;
  result(295) = select_rkind3
!  print *,"! select_rkind4"
!  print *,select_rkind4;
  result(296) = select_rkind4
!  print *,"! select_rkind5"
!  print *,select_rkind5;
  result(297) = select_rkind5

!  print *,"! select_rkind6"
!  print *,select_rkind6;
  result(298) = select_rkind6
!  print *,"! select_rkind7"
!  print *,select_rkind7;
  result(299) = select_rkind7
!  print *,"! select_rkind8"
!  print *,select_rkind8;
  result(300) = select_rkind8
!  print *,"! select_rkind9"
!  print *,1elect_rkind9;
  result(301) = select_rkind9
!  print *,"! select_rkind10"
!  print *,select_rkind10;
  result(302) = select_rkind10
!  print *,"! select_rkind11"
!  print *,select_rkind11;
  result(303) = select_rkind11
!  print *,"! select_rkind12"
!  print *,select_rkind12;
  result(304) = select_rkind12

!  print *,"! select_rkind13"
!  print *,select_rkind13;
  result(305) = select_rkind13
!  print *,"! select_rkind14"
!  print *,select_rkind14;
  result(306) = select_rkind14
!  print *,"! select_rkind15"
!  print *,select_rkind15;
  result(307) = select_rkind15
!  print *,"! select_rkind16"
!  print *,select_rkind16;
  result(308) = select_rkind16
!  print *,"! select_rkind17"
!  print *,select_rkind17;
  result(309) = select_rkind17
!  print *,"! select_rkind18"
!  print *,select_rkind18;
  result(310) = select_rkind18
!  print *,"! select_rkind19"
!  print *,select_rkind19;
  result(311) = select_rkind19
!  print *,"! select_rkind20"
!  print *,select_rkind20;
  result(312) = select_rkind20
!  print *,"! select_rkind21"
!  print *,select_rkind21;
  result(313) = select_rkind21
!  print *,"! select_rkind22"
!  print *,select_rkind22;
  result(314) = select_rkind22

!  print *,"! i2dimarryparam1"
!  print *,i2dimarryparam1;
  result(315:324) = reshape( i2dimarryparam1, (/10/))
!  print *,"! i2dimarryparam2"
!  print *,i2dimarryparam2;
  result(325:334) = reshape(i2dimarryparam2, (/10/))
!  print *,"! i2dimarryparam3"
!  print *,i2dimarryparam3;
  result(335:344) = reshape(i2dimarryparam3, (/10/))
!  print *,"! i2dimarryparam3"
!  print *,i2dimarryparam3;
  result(345:354) = reshape(i2dimarryparam3, (/10/))
!  print *,"! i2dimarryparam4"
!  print *,i2dimarryparam4;
  result(355:364) = reshape(i2dimarryparam4, (/10/))
!  print *,"! i2dimarryparam5"
!  print *,i2dimarryparam5;
  result(365:374) = reshape(i2dimarryparam5, (/10/))
!  print *,"! i2dimarryparam6"
!  print *,i2dimarryparam6;
  result(375:384) = reshape(i2dimarryparam6, (/10/))

!  print  *,"! len1"
!  print  *,len1
  result(385) = len1
!  print  *,"! len2"
!  print  *,len2
  result(386) = len2
!  print  *,"! len3"
!  print  *,len3
  result(387) = len3
!  print  *,"! len4"
!  print  *,len4
  result(388) = len4

!  print  *,"! size2"
!  print  *,size2
  result(389) = size2
!  print  *,"! size3"
!  print  *,size3
  result(390) = size3
!  print  *,"! size4"
!  print  *,size4
  result(391) = size4
!  print  *,"! size5"
!  print  *,size5
  result(392) = size5
!  print  *,"! size6"
!  print  *,size6
  result(393) = size6
!  print  *,"! size7"
!  print  *,size7
  result(394) = size7
!  print  *,"! size8"
!  print  *,size8
  result(395) = size8
!  print  *,"! size9"
!  print  *,size9
  result(396) = size9

!  print  *,"! ishft1"
!  print  *,ishft1
  result(397) = ishft1
!  print  *,"! ishft2"
!  print  *,ishft2
  result(398) = ishft2
!  print  *,"! ishftarry1"
!  print  *,ishftarry1
  result(399:403) = ishftarry1
!  print  *,"! ishftarry2"
!  print  *,ishftarry2
  result(404:408) = ishftarry2
!  print  *,"! ishftarry3"
!  print  *,ishftarry3
  result(409:413) = ishftarry3
!  print  *,"! ishftarry4"
!  print  *,ishftarry4
  result(414:418) = ishftarry4


  call check(result,expect,NTEST)

end program
subroutine copy_str_to_result( str, result)
  integer :: result(:)
  character(len=*) :: str
  integer :: i

!  print *, "str '", str, "'", ", len ", len(str)
  do i=1,len(str)
     result(i) = iachar(str(i:i))
  enddo

end subroutine
