!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!* Attribute oriented initializations using intrinsic functions

program e10
 
   parameter(NTEST=76)
   integer :: result(NTEST)
   integer :: expect(NTEST) = (/  &
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
   !char1
       65,		&
   !char2
       66,		&
   !chararry1
       48, 49, 50, 51, 52, 53,  &
    !index kind
       2,4,8,           &
    !scan kind
       2,4,8,           &
    !verify kind
       2,4,8,           &
    !len_trim kind
       2,4,8,           &
    !char kind
       1 &
   /)

   integer, parameter :: iparam = 8
   integer, parameter :: iparamarry(5) = (/(i,i=1,5)/)
   character(len=6), parameter :: strparamarry1(4) = &
                    (/"abcz  ", " abc  ", "  abc ", "   abc" /)
   character(len=6), parameter :: strparam1 = "   abc"
   character(len=2), parameter :: strparam2 = "bc"

   character(len=6), parameter :: strparam3 = "   abc"

  type t1 
    character(len=6) :: str1 	
  end type
  type (t1), parameter:: t1_param1 = t1( adjustl("   abc") )


  integer :: idx1 = index("xxyybcwwvv", "bc", KIND=4)
  integer :: idx2 = index("xxyybcwwvv", "bc", .TRUE., KIND=4)
  integer :: idx3 = index("zxxyybcwwvvzz", "bc", .FALSE., KIND=4)
  integer :: idx4 = index("zxxyybcwwvvzz", strparam2, .FALSE., KIND=4)
  integer :: idx5 = index(strparam1, "bc", KIND=4)
  integer :: idx6 = index(strparam1, "bc", .TRUE., KIND=4)
  integer :: idx7 = index(strparam1, strparam2, .TRUE., KIND=4)
  integer :: idx8 = index(strparam1, "bc", .FALSE., KIND=4)
  integer :: idx9 = index(t1_param1%str1, "bc", .FALSE., KIND=8)
  integer :: idxarry1(4) = index(strparamarry1, "bc", KIND=4)
  integer :: idxarry2(2) = index((/"abc   ", " abc  "/), "bc", KIND=4)

  integer :: scan1 = scan("xxyybcwwvv", "bc", KIND=4)
  integer :: scan2 = scan("xxyybcwwvv", "bc", .TRUE., KIND=4)
  integer :: scan3 = scan("zxxyybcwwvvzz", "bc", .FALSE., KIND=4)
  integer :: scan4 = scan("zxxyybcwwvvzz", strparam2, .FALSE., KIND=4)
  integer :: scan5 = scan(strparam1, "bc", KIND=4)
  integer :: scan6 = scan(strparam1, "bc", .TRUE.,KIND=4)
  integer :: scan7 = scan(strparam1, strparam2, .TRUE.,KIND=4)
  integer :: scan8 = scan(strparam1, "bc", .FALSE.,KIND=4)
  integer :: scan9 = scan(t1_param1%str1, "bc", .FALSE.,KIND=4)
  integer :: scanarry1(4) = scan(strparamarry1, "bc",KIND=4)
  integer :: scanarry2(2) = scan((/"abc   ", " aac  "/), "bc",KIND=4)

  integer :: vrfy1 = verify("bbbbc", "bc", KIND=4)
  integer :: vrfy2 = verify("xxyybcwwvv", "xywzbc", .TRUE.,KIND=4)
  integer :: vrfy3 = verify("zxxyybcwwvvzz", "bc", .FALSE., KIND=4)
  integer :: vrfy4 = verify("zxxyybcwwvvzz", strparam2, .FALSE., KIND=4)
  integer :: vrfy5 = verify(strparam1, "abc", KIND=4)
  integer :: vrfy6 = verify(strparam1, "bac", .TRUE.,KIND=4)
  integer :: vrfy7 = verify(strparam1, strparam2, .TRUE.,KIND=4)
  integer :: vrfy8 = verify(strparam1, "abc", .FALSE.,KIND=4)
  integer :: vrfy9 = verify(t1_param1%str1, "bc", .FALSE.,KIND=4)
  integer :: vrfyarry1(4) = verify(strparamarry1, " abc",KIND=4)
  integer :: vrfyarry2(2) = verify((/"abc   ", " abc  "/), "abc",KIND=4)

  integer :: lentrim1 = len_trim("xxyybc    ",KIND=4)
  integer :: lentrim2 = len_trim("zxxyybcwwvvzz",KIND=4)
  integer :: lentrim3 = len_trim(strparam3,KIND=4)
  integer :: lentrim4 = len_trim(t1_param1%str1,KIND=4)
  integer :: lentrimarry1(4) = len_trim(strparamarry1,KIND=4)
  integer :: lentrimarry2(2) = len_trim((/"abc   ", " abc  "/),KIND=4)

  integer, parameter :: iparam2 = 66
  character :: char1 = char(65, KIND=1)
  character :: char2 = char(iparam2, KIND=1)
  character :: chararry1(6) = char((/48,49,50,51,52,53/),KIND=1)


!  print *,"! idx1"
!  print *,idx1;
  result(1) = idx1
!  print *,"! idx2"
!  print *,idx2;
  result(2) = idx2
!  print *,"! idx3"
!  print *,idx3;
  result(3) = idx3
!  print *,"! idx4"
!  print *,idx4;
  result(4) = idx4
!  print *,"! idx5"
!  print *,idx5;
  result(5) = idx5
!  print *,"! idx6"
!  print *,idx6;
  result(6) = idx6
!  print *,"! idx7"
!  print *,idx7;
  result(7) = idx7
!  print *,"! idx8"
!  print *,idx8;
  result(8) = idx8
!  print *,"! idx9"
!  print *,idx9;
  result(9) = idx9
!  print *,"! idxarry1"
!  print *,idxarry1;
  result(10:13) = idxarry1
!  print *,"! idxarry2"
!  print *,idxarry2;
  result(14:15) = idxarry2
!  print *,"! lentrim1"
!  print *,lentrim1;
  result(16) = lentrim1
!  print *,"! lentrim2"
!  print *,lentrim2;
  result(17) = lentrim2
!  print *,"! lentrim3"
!  print *,lentrim3;
  result(18) = lentrim3
!  print *,"! lentrim4"
!  print *,lentrim4;
  result(19) = lentrim4
!  print *,"! lentrimarry1"
!  print *,lentrimarry1;
  result(20:23) = lentrimarry1
!  print *,"! lentrimarry2"
!  print *,lentrimarry2;
  result(24:25) = lentrimarry2

!  print *,"! scan1"
!  print *,scan1;
  result(26) = scan1
!  print *,"! scan2"
!  print *,scan2;
  result(27) = scan2
!  print *,"! scan3"
!  print *,scan3;
  result(28) = scan3
!  print *,"! scan4"
!  print *,scan4;
  result(29) = scan4
!  print *,"! scan5"
!  print *,scan5;
  result(30) = scan5
!  print *,"! scan6"
!  print *,scan6;
  result(31) = scan6
!  print *,"! scan7"
!  print *,scan7;
  result(32) = scan7
!  print *,"! scan8"
!  print *,scan8;
  result(33) = scan8
!  print *,"! scan9"
!  print *,scan9;
  result(34) = scan9
!  print *,"! scanarry1"
!  print *,scanarry1;
  result(35:38) = scanarry1
!  print *,"! scanarry2"
!  print *,scanarry2;
  result(39:40) = scanarry2

!  print *,"! vrfy1"
!  print *,vrfy1;
  result(41) = vrfy1
!  print *,"! vrfy2"
!  print *,vrfy2;
  result(42) = vrfy2
!  print *,"! vrfy3"
!  print *,vrfy3;
  result(43) = vrfy3
!  print *,"! vrfy4"
!  print *,vrfy4;
  result(44) = vrfy4
!  print *,"! vrfy5"
!  print *,vrfy5;
  result(45) = vrfy5
!  print *,"! vrfy6"
!  print *,vrfy6;
  result(46) = vrfy6
!  print *,"! vrfy7"
!  print *,vrfy7;
  result(47) = vrfy7
!  print *,"! vrfy8"
!  print *,vrfy8;
  result(48) = vrfy8
!  print *,"! vrfy9"
!  print *,vrfy9;
  result(49) = vrfy9
!  print *,"! vrfyarry1"
!  print *,vrfyarry1;
  result(50:53) = vrfyarry1
!  print *,"! vrfyarry2"
!  print *,vrfyarry2;
  result(54:55) = vrfyarry2

!   print *, char1
  result(56) = ichar(char1)
!   print *, char2 
  result(57) =ichar(char2)
!   print *, chararry1(1)
!   print *, chararry1(2)
!   print *, chararry1(3)
!   print *, chararry1(4)
!   print *, chararry1(5)
!   print *, chararry1(6)
  result(58:63) =ichar(chararry1)

!check kind
result(64) = kind(index("xxyybcwwvv", "bc", KIND=2));
!print *, result(64)
result(65) = kind(index("xxyybcwwvv", "bc", KIND=4));
!print *, result(65)
result(66) = kind(index("xxyybcwwvv", "bc", KIND=8));
!print *, result(66)
result(67) = kind(scan("xxyybcwwvv", "bc", KIND=2))
!print *, result(67)
result(68) = kind(scan("xxyybcwwvv", "bc", KIND=4))
!print *, result(68)
result(69) = kind(scan("xxyybcwwvv", "bc", KIND=8))
!print *, result(69)
result(70) = kind(verify("bbbbc", "bc", KIND=2))
!print *, result(70)
result(71) = kind(verify("bbbbc", "bc", KIND=4))
!print *, result(71)
result(72) = kind(verify("bbbbc", "bc", KIND=8))
!print *, result(72)
result(73) = kind(len_trim("xxyybc    ",KIND=2))
!print *, result(73)
result(74) = kind(len_trim("xxyybc    ",KIND=4))
!print *, result(74)
result(75) = kind(len_trim("xxyybc    ",KIND=8))
!print *, result(75)
result(76) = kind(char(65, KIND=1))
!print *, result(76)

  call check(result,expect,NTEST)

end program
