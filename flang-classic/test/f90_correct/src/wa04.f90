
!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!  Test entity oriented variable initializations
!

module m
 INTEGER, parameter :: intParam1 = 2
 !INTEGER, parameter :: intParam2 = intParam1+3
 INTEGER, parameter :: intParamArr4(2)= (/3,4/)
 INTEGER, parameter :: intParamArr5(2)= intParamArr4

 INTEGER :: int1
 INTEGER :: intarr1(2)

 type t1
   integer :: i
   integer :: j
   integer :: k
 end type
 type (t1):: t1_inst
 type (t1), parameter :: t1_param1 = t1(1,2,3)
 type (t1), parameter :: t1_param2 = t1_param1
 type (t1), parameter :: t1_param_array(3) = (/t1(1,2,3), t1(2,3,4), t1(3,4,5)/)

 data int1,intarr1,t1_inst / 18, 81, 82, t1(1,2,3) /
end module m


SUBROUTINE module_test(strt_idx, result)
  use m

 INTEGER :: strt_idx 
 INTEGER, dimension(:) :: result 

 type t2
   integer :: i
   integer :: j
   integer :: k
 end type
 type (t2):: t2_inst = t2(6,7,8)
!
 result(strt_idx) = intParam1
 result(strt_idx+1:strt_idx+2) = intParamArr4
 result(strt_idx+3:strt_idx+4) = intParamArr5
 result(strt_idx+5) = int1
 result(strt_idx+6:strt_idx+7) = intarr1
 result(strt_idx+8) = t1_inst%i
 result(strt_idx+9) = t1_inst%j
 result(strt_idx+10) = t1_inst%k
 result(strt_idx+11) = t2_inst%i
 result(strt_idx+12) = t2_inst%j
 result(strt_idx+13) = t2_inst%k

END SUBROUTINE


PROGRAM wa04
!
 PARAMETER (N=247)

 INTEGER, dimension(N) :: result, expect

 INTERFACE
   SUBROUTINE module_test( strt_idx, result)
     INTEGER :: strt_idx
     INTEGER, dimension(:) :: result
   END SUBROUTINE
 END INTERFACE

 data expect / &
! intParam1
  2, &
! intParam2
  5, &
! int8Param1
  16, &
! intParamArr1
  6,6,6,6,6,6,6,6,6,6, &
! intParamArr2
  3,3,3,3,3,3,3,3,3,3, &
! intParamArr3
  3,4, &
! intParamArr4
  3,4, &
! intParamArr5
  6,8, &
! intParamArr6
  12,12,12,12,12,12,12,12,12,12, &
! intParamArr7
  9,16, &
! intParamArr8
  6,20, &
! intParamArr9
  5,32, &
! int1
  2, &
! int2
  15, &
! int3
  3,4, &
! int4
  33, &
! intArr1
  5,6, &
! intArr2
  6,8, &
! intArr3
  3,5,7,9,11, &
! intArr4
  2,3,7,8,9, &
! intArr5
  6,6,6,6,6, &
! intArr6
  6,8, &
! intArr7
  2,2,2,2,2,2,2,2,2,2, &
! intArr8
  1,2,3,4,5,6,7,8,9,10, &
! intArr9
  20,20,20,20,20,20,20,20,20,20, &
! intArr10
  3,4, &
! intArr11
  -2,-2,-2,-2,-2,-2,-2,-2,-2,-2, &
! intArr12
  51,42,33,24,15, &
! t1_param1
  1,2,3, &
! t1_param2
  1,2,3, &
! t1_param_array
  1,2,3,2,3,4,3,4,5, &
! t1_inst1
  2,2,3, &
! t1_inst2
  1,2,3, &
! t1_array
  1,2,3,2,3,4,3,4,5, &
! t2param1
  4,51,42,33,24,15, &
! t2inst1
  1,1,2,3,4,5, &
! t2inst2
  2,3,4,5,6,7, &
! t2inst3
  2,51,42,33,24,15, &
! t3inst1
  2,4,2,0, &
! t3inst2
  2,1,2,3, &
! t4inst1
  2,4,2,0,8, &
! t4inst2
  2,1,2,3,7, &
! t4inst3
  2,1,2,3,5, &
! t5inst1
  4,2,0,2, &
! t5inst2
  1,2,3,2, &
! t6Param1
  4,51,42,33,24,15,2, &
! t3inst3
  1,1,2,3, &
! t3inst4
  6,1,2,3, &
! t7inst1
  1,2,3, &
! t6inst1
  4,51,42,33,24,15,1, &
! FROM module_test
! module m: intParam1
  2, &
! module m: intParamArr4
  3,4, &
! module m: intParamArr5
  3,4, &
! module m: int1
  18, &
! module m: int2
  81,82, &
! module m: t1_inst
  1,2,3, &
!module m:  t2_inst
  6,7,8, &
!t8_param
  3,4,5,6,7 /

 INTEGER, parameter :: intParam1 = 2
 INTEGER, parameter :: intParam2 = intParam1+3
 INTEGER*8, parameter :: int8Param1 = 16
 INTEGER, parameter :: intParamArr1(10) = 6
 INTEGER, parameter :: intParamArr2(10)= (/(3,i=1,10)/)
 INTEGER, parameter :: intParamArr3(2)= (/3,4/)
 INTEGER, parameter :: intParamArr4(2)= (/3.0,4.0/)
 INTEGER, parameter :: intParamArr5(2)= (/3.0,4.0/)*2
 INTEGER, parameter :: intParamArr6(10)= intParamArr1*2
! constant OP_XTOI/OP_XTOK
 INTEGER, parameter :: intParamArr7(2)= (/3,4/)**2
 INTEGER, parameter :: intParamArr8(2)= (/2.5,4.5/)**2.0
! constant OP_XTOX
 INTEGER, parameter :: intParamArr9(2)= (/2.0,4.0/)**2.5

 INTEGER :: int1 = intParam1
 INTEGER :: int2 = (intParam1+1)*intParam2
 INTEGER :: int3(2)=  intParamArr3

 INTEGER :: intArr1(2)= (/5,6/)
 INTEGER :: intArr2(2) = (/ intParamArr3*2 /)
 INTEGER :: intArr3(5)  = (/ (iii+2,iii=1,10,2) /)
 INTEGER :: intArr4(5)  = (/ intParam1, 3, 7, 8 ,9/)
 INTEGER :: intArr5(5)  = (/ intParamArr1(2:6) /)
 INTEGER :: intArr6(2) = ((/1,2/)+2)*intParam1
 INTEGER :: intArr7(10) = intParam1
 INTEGER :: intArr8(10)= (/(iii,iii=1,10)/)
 INTEGER :: intArr9(10) = 10*intparam1
 INTEGER :: intArr10(2) = (/1,2/)+2
 INTEGER :: intArr11(10) = -intparam1
!
 type t1
    integer :: i
    integer :: j
    integer :: k
 end type
 type (t1), parameter :: t1_param1 = t1(1,2,3)
 type (t1), parameter :: t1_param2 = t1_param1
 type (t1), parameter :: t1_param_array(3) = (/t1(1,2,3), t1(2,3,4), t1(3,4,5)/)
 
 type (t1) :: t1_inst1 = t1(intParam1,2,3)
 type (t1) :: t1_inst2 = t1_param1;
 type (t1) :: t1_array(3) = t1_param_array
!
 type t2
    integer :: i
    integer :: iary(1:5)
 end type
!
 type (t2), parameter :: t2param1 = t2(4, (/51,42,33,24,15/))
 type (t2) :: t2inst1 = t2(1, (/1,2,3,4,5/))
 type (t2) :: t2inst2 = t2(intParam1, (/(i+2,i=1,5)/) )
 type (t2) :: t2inst3 = t2(intParam1, t2param1%iary )
 INTEGER :: int4 = t2param1%iary(3)		
!
 type t3
    integer :: j
    type(t1) :: t1_inst
 end type
!
 type(t3) :: t3inst1 = t3(intParam1, t1(4,2,0) )
 type(t3) :: t3inst2 = t3(intParam1, t1_param1)
 type(t3) :: t3inst3 = t3(t1_param1%i, t1_param1)
 type(t3) :: t3inst4 = t3(intParamArr1(2:2), t1_param1)
!
 type t4
    integer :: j
    type(t1) :: t1_inst
    integer :: k
 end type
!
 type(t4) :: t4inst1 = t4(intParam1, t1(4,2,0), 8)
 type(t4) :: t4inst2 = t4(intParam1, t1_param1, 7)
 type(t4) :: t4inst3 = t4(intParam1, t1_param1, intParam2)
!
 type t5
    type(t1) :: t1_inst
    integer :: j
 end type
!
 type(t5) :: t5inst1 = t5(t1(4,2,0), intParam1)
 type(t5) :: t5inst2 = t5(t1_param1, intParam1)
!
 type t6
    type(t2) :: t2_inst
    integer :: j
 end type
!
 type(t6), parameter :: t6Param1 = t6(t2(4, (/51,42,33,24,15/)) ,2)
 type(t6) :: t6inst1 = t6(t6Param1%t2_inst,1)
 INTEGER :: intArr12(1:5) = t6Param1%t2_inst%iary
!
 type t7
    type(t1) :: t1_inst
 end type
 type(t7) :: t7inst1 = t7(t1_param1)

 type t8
   integer :: t8_intArr(5)
 end type
 type(t8), parameter :: t8_param = t8( (/1,2,3,4,5/) + 2 )



!
 result(1) = intParam1
 result(2) = intParam2
 result(3) = int8Param1
 result(4:13) = intParamArr1
 result(14:23) = intParamArr2
 result(24:25) = intParamArr3
 result(26:27) = intParamArr4
 result(28:29) = intParamArr5
 result(30:39) = intParamArr6
 result(40:41) = intParamArr7
 result(42:43) = intParamArr8
 result(44:45) = intParamArr9
 result(46) = int1
 result(47) = int2
 result(48:49) = int3
 result(50) = int4
 result(51:52) = intArr1
 result(53:54) = intArr2
 result(55:59) = intArr3
 result(60:64) = intArr4
 result(65:69) = intArr5
 result(70:71) = intArr6
 result(72:81) = intArr7
 result(82:91) = intArr8
 result(92:101) = intArr9
 result(102:103) = intArr10
 result(104:113) = intArr11
 result(114:118) = intArr12
 result(119) = t1_param1%i
 result(120) = t1_param1%j
 result(121) = t1_param1%k
 result(122) = t1_param2%i
 result(123) = t1_param2%j
 result(124) = t1_param2%k
 result(125) = t1_param_array(1)%i
 result(126) = t1_param_array(1)%j
 result(127) = t1_param_array(1)%k
 result(128) = t1_param_array(2)%i
 result(129) = t1_param_array(2)%j
 result(130) = t1_param_array(2)%k
 result(131) = t1_param_array(3)%i
 result(132) = t1_param_array(3)%j
 result(133) = t1_param_array(3)%k
 result(134) = t1_inst1%i
 result(135) = t1_inst1%j
 result(136) = t1_inst1%k
 result(137) = t1_inst2%i
 result(138) = t1_inst2%j
 result(139) = t1_inst2%k
 result(140) = t1_array(1)%i
 result(141) = t1_array(1)%j
 result(142) = t1_array(1)%k
 result(143) = t1_array(2)%i
 result(144) = t1_array(2)%j
 result(145) = t1_array(2)%k
 result(146) = t1_array(3)%i
 result(147) = t1_array(3)%j
 result(148) = t1_array(3)%k
 result(149) = t2param1%i
 result(150) = t2param1%iary(1)
 result(151) = t2param1%iary(2)
 result(152) = t2param1%iary(3)
 result(153) = t2param1%iary(4)
 result(154) = t2param1%iary(5)
 result(155) = t2inst1%i
 result(156) = t2inst1%iary(1)
 result(157) = t2inst1%iary(2)
 result(158) = t2inst1%iary(3)
 result(159) = t2inst1%iary(4)
 result(160) = t2inst1%iary(5)
 result(161) = t2inst2%i
 result(162) = t2inst2%iary(1)
 result(163) = t2inst2%iary(2)
 result(164) = t2inst2%iary(3)
 result(165) = t2inst2%iary(4)
 result(166) = t2inst2%iary(5)
 result(167) = t2inst3%i
 result(168) = t2inst3%iary(1)
 result(169) = t2inst3%iary(2)
 result(170) = t2inst3%iary(3)
 result(171) = t2inst3%iary(4)
 result(172) = t2inst3%iary(5)
 result(173) = t3inst1%j
 result(174) = t3inst1%t1_inst%i
 result(175) = t3inst1%t1_inst%j
 result(176) = t3inst1%t1_inst%k
 result(177) = t3inst2%j
 result(178) = t3inst2%t1_inst%i
 result(179) = t3inst2%t1_inst%j
 result(180) = t3inst2%t1_inst%k
 result(181) = t4inst1%j
 result(182) = t4inst1%t1_inst%i
 result(183) = t4inst1%t1_inst%j
 result(184) = t4inst1%t1_inst%k
 result(185) = t4inst1%k
 result(186) = t4inst2%j
 result(187) = t4inst2%t1_inst%i
 result(188) = t4inst2%t1_inst%j
 result(189) = t4inst2%t1_inst%k
 result(190) = t4inst2%k
 result(191) = t4inst3%j
 result(192) = t4inst3%t1_inst%i
 result(193) = t4inst3%t1_inst%j
 result(194) = t4inst3%t1_inst%k
 result(195) = t4inst3%k
 result(196) = t5inst1%t1_inst%i
 result(197) = t5inst1%t1_inst%j
 result(198) = t5inst1%t1_inst%k
 result(199) = t5inst1%j
 result(200) = t5inst2%t1_inst%i
 result(201) = t5inst2%t1_inst%j
 result(202) = t5inst2%t1_inst%k
 result(203) = t5inst2%j
 result(204) = t6Param1%t2_inst%i
 result(205) = t6Param1%t2_inst%iary(1)
 result(206) = t6Param1%t2_inst%iary(2)
 result(207) = t6Param1%t2_inst%iary(3)
 result(208) = t6Param1%t2_inst%iary(4)
 result(209) = t6Param1%t2_inst%iary(5)
 result(210) = t6Param1%j
 result(211) = t3inst3%j
 result(212) = t3inst3%t1_inst%i
 result(213) = t3inst3%t1_inst%j
 result(214) = t3inst3%t1_inst%k
 result(215) = t3inst4%j
 result(216) = t3inst4%t1_inst%i
 result(217) = t3inst4%t1_inst%j
 result(218) = t3inst4%t1_inst%k
 result(219:) = t7inst1%t1_inst%i
 result(220:) = t7inst1%t1_inst%j
 result(221:) = t7inst1%t1_inst%k
 result(222) = t6inst1%t2_inst%i
 result(223) = t6inst1%t2_inst%iary(1)
 result(224) = t6inst1%t2_inst%iary(2)
 result(225) = t6inst1%t2_inst%iary(3)
 result(226) = t6inst1%t2_inst%iary(4)
 result(227) = t6inst1%t2_inst%iary(5)
 result(228) = t6inst1%j

 call module_test(229,result)
 result(243:247) = t8_param%t8_intArr;
!
 call check(result, expect, N);

end program
