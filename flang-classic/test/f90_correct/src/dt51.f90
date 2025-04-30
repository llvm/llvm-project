!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

  program dt51

    parameter(N=676)

    INTEGER :: result(N)
    INTEGER :: expect(N)

    data expect / &
    ! tda_param1:
       2, 98, 42, 22, 11, &
    ! tda_inst:
        0, 97, 3, 47, 52, &
    ! tda_array1:
        0, 97, 3, 47, 52, 0, 97, 3, 47, 52, 0, 97, 3, 47, 52, &
    ! tda_inst_ptr1:
        0, 97, 3, 47, 52, &
    ! tda_inst_ptr2:
        0, 97, 3, 47, 52, &
    ! tde_inst%tda_ptr:
        0, 97, 3, 47, 52, &
    ! tdb_inst_ptr1:
        0, 0, 97, 3, 47, 52, 2, 97, 42, 43, 44, 0, 97, &
        3, 47, 52, 0, 97, 3, 47, 52, 0, 97, 3, 47, &
        52, 2, 98, 42, 22, 11, 2, 98, 42, 22, 11, 2, 98, &
        42, 22, 11, 1, 98, 42, 44, 43, 1, 98, 42, 44, &
        43, 1, 98, 42, 44, 43, 1, 98, 42, 11, 22, 1, 98, &
        42, 11, 22, 1, 98, 42, 11, 22, 1, 98, 42, 11, &
        22, 1, 98, 42, 11, 22, 1, 98, 42, 11, 22, 0, &
    ! tdc_inst_ptr1:
        0, 0, 97, 3, 47, 52, 2, 97, 42, 6, 7, 2, 98, &
        42, 22, 11, 0, 0, 97, 3,47, 52, 2, 97, 42, &
        43, 44, 0, 97, 3, 47, 52, 0, 97, 3, 47, 52, &
        0, 97, 3, 47, 52, 2, 98, 42, 22, 11, 2, 98, 42, &
        22, 11, 2, 98, 42, 22, 11, 1, 98, 42, 44, 43, &
        1, 98, 42, 44, 43, 1, 98, 42, 44, 43, 1, 98, 42, &
        11, 22, 1, 98, 42, 11, 22, 1, 98, 42, 11, 22, &
        1, 98, 42, 11, 22, 1, 98, 42, 11, 22, 1, 98, 42, &
        11, 22, 0, 0, &
    ! tdb_alloc_array1:
        0, 0, 97, 3, 47, 52, 2, 97, 42, 43, 44, 0, 97, &
        3, 47, 52, 0, 97, 3, 47, 52, 0, 97, 3, 47, &
        52, 2, 98, 42, 22, 11, 2, 98, 42, 22, 11, 2, 98, &
        42, 22, 11, 1, 98, 42, 44, 43, 1, 98, 42, 44, &
        43, 1, 98, 42, 44, 43, 1, 98, 42, 11, 22, 1, 98, &
        42, 11, 22, 1, 98, 42, 11, 22, 1, 98, 42, 11, &
        22, 1, 98, 42, 11, 22, 1, 98, 42, 11, 22, 0, &
        0, 0, 97, 3, 47, 52, 2, 97, 42, 43, 44, 0, 97, &
        3, 47, 52, 0, 97, 3, 47, 52, 0, 97, 3, 47, &
        52, 2, 98, 42, 22, 11, 2, 98, 42, 22, 11, 2, 98, &
        42, 22, 11, 1, 98, 42, 44, 43, 1, 98, 42, 44, &
        43, 1, 98, 42, 44, 43, 1, 98, 42, 11, 22, 1, 98, &
        42, 11, 22, 1, 98, 42, 11, 22, 1, 98, 42, 11, &
        22, 1, 98, 42, 11, 22, 1, 98, 42, 11, 22, 0, &
    ! tdb_inst:
        0, 0, 97, 3, 47, 52, 2, 97, 42, 43, 44, 0, 97, &
        3, 47, 52, 0, 97, 3, 47, 52, 0, 97, 3, 47, &
        52, 2, 98, 42, 22, 11, 2, 98, 42, 22, 11, 2, 98, &
        42, 22, 11, 1, 98, 42, 44, 43, 1, 98, 42, 44, &
        43, 1, 98, 42, 44, 43, 1, 98, 42, 11, 22, 1, 98, &
        42, 11, 22, 1, 98, 42, 11, 22, 1, 98, 42, 11, &
        22, 1, 98, 42, 11, 22, 1, 98, 42, 11, 22, 0, &
    ! tdc_inst:
        0, 0, 97, 3, 47, 52, 2, 97, 42, 6, 7, 2, 98, &
        42, 22, 11, 0, 0, 97, 3, 47, 52, 2, 97, 42, &
        43, 44, 0, 97, 3, 47, 52, 0, 97, 3, 47, 52, &
        0, 97, 3, 47, 52, 2, 98, 42, 22, 11, 2, 98, 42, &
        22, 11, 2, 98, 42, 22, 11, 1, 98, 42, 44, 43, &
        1, 98, 42, 44, 43, 1, 98, 42, 44, 43, 1, 98, 42, &
        11, 22, 1, 98, 42, 11, 22, 1, 98, 42, 11, 22, &
        1, 98, 42, 11, 22, 1, 98, 42, 11, 22, 1, 98, 42, &
        11, 22, 0, 0, &
    ! tdd_inst:
        2, 4, 5, 6, 7, 8, 43, 43, 43, 43, 43, 3, &
        5, 7, 9, 11, 16, 17, 18, 16, 17, 18, 32, 34, &
        36, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 2021, &
        2021, 2021, 2021, 2021, &
    ! tdd_ptr1:
        2, 4, 5, 6, 7, 8, 43, 43, 43, 43, 43, 3, &
        5, 7, 9, 11, 16, 17, 18, 16, 17, 18, 32, 34, &
        36, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 2021, &
        2021, 2021, 2021, 2021/

    integer, parameter :: intParam1 = 47
    integer, parameter :: intParamArr1(3) = (/ 16,17,18 /)
    integer, parameter :: intParamArr2(3) = intParamArr1
    integer, parameter :: intParamArr3(3) = (/intParamArr1/)
    type tda
      integer :: a1
      character*4 :: a2 = 'a'
      integer :: a3 = 3
      integer :: a4 = intParam1
      integer :: a5 = intParam1 + 5
    end type
    type(tda), parameter ::tda_param1 =  tda(2,'b',42,22,11)
    type(tda), parameter ::tda_param_arr1(3) =  tda(1,'b',42,11,22)
    type(tda), parameter ::tda_param_arr3(3) =  tda_param_arr1
    type(tda), parameter ::tda_param_arr4(3) = (/tda_param_arr1/)
    type(tda)::tda_inst
    type(tda)::tda_array1(3)
    type(tda), pointer::tda_inst_ptr1
    type(tda), pointer::tda_inst_ptr2

    type tdb
      integer :: b1
      type(tda) :: tda_mbr_inst1
      type(tda) :: tda_mbr_inst2 = tda(2,'a',42,43,44)
      type(tda) :: tda_mbr_array1(3)
      type(tda) :: tda_mbr_array2(3) = tda_param1
      type(tda) :: tda_mbr_array4(3) =   tda(1,'b',42,44,43)
      type(tda) :: tda_mbr_array5(3) = tda_param_arr1
      type(tda) :: tda_mbr_array6(3) = (/ tda_param_arr1 /)
      integer :: b2
    end type
    type (tdb) :: tdb_inst
    type (tdb), pointer :: tdb_inst_ptr1
    type (tdb), allocatable, dimension(:) :: tdb_alloc_array1

    type tdc
      integer :: b1
      type(tda) :: tda_inst_mbr1
      type(tda) :: tda_inst_mbr2 = tda(2,'a',42,6,7)
      type(tda) :: tda_inst_mbr3 = tda_param1
      type(tdb) :: tdb_inst_mbr1
      integer :: b2
    end type
    type (tdc) :: tdc_inst
    type (tdc), pointer :: tdc_inst_ptr1

    type tdd
      integer :: tdd_int1 = tda_param1%a1
      integer :: tdd_intArray1(5) = (/ 1,2,3,4,5/)+3
      integer :: tdd_intArray2(5) = 43
      integer :: tdd_intArray3(5) = (/  (iii+2,iii=1,10,2) /)
      integer :: tdd_intArray4(3) = intParamArr1
      integer :: tdd_intArray5(3) = (/ intParamArr1 /)
      integer :: tdd_intArray6(3) = (/ intParamArr1 /)*2
      integer :: tdd_intArray7(5) = (/ (ii, ii=1,5) /)
      integer :: tdd_intArray8(5) = (/ (ii+2, ii=1,5) /) + 2
      integer :: tdd_intArray9(5) = 43 * intParam1
    end type
    type (tdd) :: tdd_inst
    type (tdd), pointer :: tdd_ptr1

    type tde
     type(tda), pointer :: tda_ptr
    end type
    type (tde) :: tde_inst

    result(1) = tda_param1%a1
    result(2) = ICHAR( tda_param1%a2)
    result(3) = tda_param1%a3
    result(4) = tda_param1%a4
    result(5) = tda_param1%a5

    result(6) = tda_inst%a1
    result(7) = ICHAR( tda_inst%a2)
    result(8) = tda_inst%a3
    result(9) = tda_inst%a4
    result(10) = tda_inst%a5

    result(11) = tda_array1(1)%a1
    result(12) = ICHAR( tda_array1(1)%a2)
    result(13) = tda_array1(1)%a3
    result(14) = tda_array1(1)%a4
    result(15) = tda_array1(1)%a5

    result(16) = tda_array1(2)%a1
    result(17) = ICHAR( tda_array1(2)%a2)
    result(18) = tda_array1(2)%a3
    result(19) = tda_array1(2)%a4
    result(20) = tda_array1(2)%a5

    result(21) = tda_array1(3)%a1
    result(22) = ICHAR( tda_array1(3)%a2)
    result(23) = tda_array1(3)%a3
    result(24) = tda_array1(3)%a4
    result(25) = tda_array1(3)%a5
!
    allocate(tda_inst_ptr1, tda_inst_ptr2, tde_inst%tda_ptr);
    result(26) = tda_inst_ptr1%a1
    result(27) = ICHAR( tda_inst_ptr1%a2)
    result(28) = tda_inst_ptr1%a3
    result(29) = tda_inst_ptr1%a4
    result(30) = tda_inst_ptr1%a5

    result(31) = tda_inst_ptr2%a1
    result(32) = ICHAR( tda_inst_ptr2%a2)
    result(33) = tda_inst_ptr2%a3
    result(34) = tda_inst_ptr2%a4
    result(35) = tda_inst_ptr2%a5

    result(36) = tde_inst%tda_ptr%a1
    result(37) = ICHAR( tde_inst%tda_ptr%a2)
    result(38) = tde_inst%tda_ptr%a3
    result(39) = tde_inst%tda_ptr%a4
    result(40) = tde_inst%tda_ptr%a5

    allocate(tdb_inst_ptr1);
    result(41) = tdb_inst_ptr1%b1
    result(42) = tdb_inst_ptr1%tda_mbr_inst1%a1
    result(43) = ICHAR(tdb_inst_ptr1%tda_mbr_inst1%a2)
    result(44) = tdb_inst_ptr1%tda_mbr_inst1%a3
    result(45) = tdb_inst_ptr1%tda_mbr_inst1%a4
    result(46) = tdb_inst_ptr1%tda_mbr_inst1%a5

    result(47) = tdb_inst_ptr1%tda_mbr_inst2%a1
    result(48) = ICHAR(tdb_inst_ptr1%tda_mbr_inst2%a2)
    result(49) = tdb_inst_ptr1%tda_mbr_inst2%a3
    result(50) = tdb_inst_ptr1%tda_mbr_inst2%a4
    result(51) = tdb_inst_ptr1%tda_mbr_inst2%a5

    result(52) = tdb_inst_ptr1%tda_mbr_array1(1)%a1
    result(53) = ICHAR( tdb_inst_ptr1%tda_mbr_array1(1)%a2)
    result(54) = tdb_inst_ptr1%tda_mbr_array1(1)%a3
    result(55) = tdb_inst_ptr1%tda_mbr_array1(1)%a4
    result(56) = tdb_inst_ptr1%tda_mbr_array1(1)%a5

    result(57) = tdb_inst_ptr1%tda_mbr_array1(2)%a1
    result(58) = ICHAR( tdb_inst_ptr1%tda_mbr_array1(2)%a2)
    result(59) = tdb_inst_ptr1%tda_mbr_array1(2)%a3
    result(60) = tdb_inst_ptr1%tda_mbr_array1(2)%a4
    result(61) = tdb_inst_ptr1%tda_mbr_array1(2)%a5

    result(62) = tdb_inst_ptr1%tda_mbr_array1(3)%a1
    result(63) = ICHAR( tdb_inst_ptr1%tda_mbr_array1(3)%a2)
    result(64) = tdb_inst_ptr1%tda_mbr_array1(3)%a3
    result(65) = tdb_inst_ptr1%tda_mbr_array1(3)%a4
    result(66) = tdb_inst_ptr1%tda_mbr_array1(3)%a5

    result(67) = tdb_inst_ptr1%tda_mbr_array2(1)%a1
    result(68) = ICHAR( tdb_inst_ptr1%tda_mbr_array2(1)%a2)
    result(69) = tdb_inst_ptr1%tda_mbr_array2(1)%a3
    result(70) = tdb_inst_ptr1%tda_mbr_array2(1)%a4
    result(71) = tdb_inst_ptr1%tda_mbr_array2(1)%a5

    result(72) = tdb_inst_ptr1%tda_mbr_array2(2)%a1
    result(73) = ICHAR( tdb_inst_ptr1%tda_mbr_array2(2)%a2)
    result(74) = tdb_inst_ptr1%tda_mbr_array2(2)%a3
    result(75) = tdb_inst_ptr1%tda_mbr_array2(2)%a4
    result(76) = tdb_inst_ptr1%tda_mbr_array2(2)%a5

    result(77) = tdb_inst_ptr1%tda_mbr_array2(3)%a1
    result(78) = ICHAR( tdb_inst_ptr1%tda_mbr_array2(3)%a2)
    result(79) = tdb_inst_ptr1%tda_mbr_array2(3)%a3
    result(80) = tdb_inst_ptr1%tda_mbr_array2(3)%a4
    result(81) = tdb_inst_ptr1%tda_mbr_array2(3)%a5

    result(82) = tdb_inst_ptr1%tda_mbr_array4(1)%a1
    result(83) = ICHAR( tdb_inst_ptr1%tda_mbr_array4(1)%a2)
    result(84) = tdb_inst_ptr1%tda_mbr_array4(1)%a3
    result(85) = tdb_inst_ptr1%tda_mbr_array4(1)%a4
    result(86) = tdb_inst_ptr1%tda_mbr_array4(1)%a5

    result(87) = tdb_inst_ptr1%tda_mbr_array4(2)%a1
    result(88) = ICHAR( tdb_inst_ptr1%tda_mbr_array4(2)%a2)
    result(89) = tdb_inst_ptr1%tda_mbr_array4(2)%a3
    result(90) = tdb_inst_ptr1%tda_mbr_array4(2)%a4
    result(91) = tdb_inst_ptr1%tda_mbr_array4(2)%a5

    result(92) = tdb_inst_ptr1%tda_mbr_array4(3)%a1
    result(93) = ICHAR( tdb_inst_ptr1%tda_mbr_array4(3)%a2)
    result(94) = tdb_inst_ptr1%tda_mbr_array4(3)%a3
    result(95) = tdb_inst_ptr1%tda_mbr_array4(3)%a4
    result(96) = tdb_inst_ptr1%tda_mbr_array4(3)%a5

    result(97) = tdb_inst_ptr1%tda_mbr_array5(1)%a1
    result(98) = ICHAR( tdb_inst_ptr1%tda_mbr_array5(1)%a2)
    result(99) = tdb_inst_ptr1%tda_mbr_array5(1)%a3
    result(100) = tdb_inst_ptr1%tda_mbr_array5(1)%a4
    result(101) = tdb_inst_ptr1%tda_mbr_array5(1)%a5

    result(102) = tdb_inst_ptr1%tda_mbr_array5(2)%a1
    result(103) = ICHAR( tdb_inst_ptr1%tda_mbr_array5(2)%a2)
    result(104) = tdb_inst_ptr1%tda_mbr_array5(2)%a3
    result(105) = tdb_inst_ptr1%tda_mbr_array5(2)%a4
    result(106) = tdb_inst_ptr1%tda_mbr_array5(2)%a5

    result(107) = tdb_inst_ptr1%tda_mbr_array5(3)%a1
    result(108) = ICHAR( tdb_inst_ptr1%tda_mbr_array5(3)%a2)
    result(109) = tdb_inst_ptr1%tda_mbr_array5(3)%a3
    result(110) = tdb_inst_ptr1%tda_mbr_array5(3)%a4
    result(111) = tdb_inst_ptr1%tda_mbr_array5(3)%a5

    result(112) = tdb_inst_ptr1%tda_mbr_array6(1)%a1
    result(113) = ICHAR( tdb_inst_ptr1%tda_mbr_array6(1)%a2)
    result(114) = tdb_inst_ptr1%tda_mbr_array6(1)%a3
    result(115) = tdb_inst_ptr1%tda_mbr_array6(1)%a4
    result(116) = tdb_inst_ptr1%tda_mbr_array6(1)%a5

    result(117) = tdb_inst_ptr1%tda_mbr_array6(2)%a1
    result(118) = ICHAR( tdb_inst_ptr1%tda_mbr_array6(2)%a2)
    result(119) = tdb_inst_ptr1%tda_mbr_array6(2)%a3
    result(120) = tdb_inst_ptr1%tda_mbr_array6(2)%a4
    result(121) = tdb_inst_ptr1%tda_mbr_array6(2)%a5

    result(122) = tdb_inst_ptr1%tda_mbr_array6(3)%a1
    result(123) = ICHAR( tdb_inst_ptr1%tda_mbr_array6(3)%a2)
    result(124) = tdb_inst_ptr1%tda_mbr_array6(3)%a3
    result(125) = tdb_inst_ptr1%tda_mbr_array6(3)%a4
    result(126) = tdb_inst_ptr1%tda_mbr_array6(3)%a5
    result(127) = tdb_inst_ptr1%b2

    allocate(tdc_inst_ptr1);

    result(128) = tdc_inst_ptr1%b1
    result(129) = tdc_inst_ptr1%tda_inst_mbr1%a1
    result(130) = ICHAR( tdc_inst_ptr1%tda_inst_mbr1%a2)
    result(131) = tdc_inst_ptr1%tda_inst_mbr1%a3
    result(132) = tdc_inst_ptr1%tda_inst_mbr1%a4
    result(133) = tdc_inst_ptr1%tda_inst_mbr1%a5

    result(134) = tdc_inst_ptr1%tda_inst_mbr2%a1
    result(135) = ICHAR( tdc_inst_ptr1%tda_inst_mbr2%a2)
    result(136) = tdc_inst_ptr1%tda_inst_mbr2%a3
    result(137) = tdc_inst_ptr1%tda_inst_mbr2%a4
    result(138) = tdc_inst_ptr1%tda_inst_mbr2%a5

    result(139) = tdc_inst_ptr1%tda_inst_mbr3%a1
    result(140) = ICHAR( tdc_inst_ptr1%tda_inst_mbr3%a2)
    result(141) = tdc_inst_ptr1%tda_inst_mbr3%a3
    result(142) = tdc_inst_ptr1%tda_inst_mbr3%a4
    result(143) = tdc_inst_ptr1%tda_inst_mbr3%a5

    result(144) = tdc_inst_ptr1%tdb_inst_mbr1%b1
    result(145) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_inst1%a1
    result(146) = ICHAR(tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_inst1%a2)
    result(147) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_inst1%a3
    result(148) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_inst1%a4
    result(149) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_inst1%a5

    result(150) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_inst2%a1
    result(151) = ICHAR(tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_inst2%a2)
    result(152) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_inst2%a3
    result(153) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_inst2%a4
    result(154) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_inst2%a5

    result(155) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array1(1)%a1
    result(156) = ICHAR( tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array1(1)%a2)
    result(157) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array1(1)%a3
    result(158) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array1(1)%a4
    result(159) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array1(1)%a5

    result(160) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array1(2)%a1
    result(161) = ICHAR( tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array1(2)%a2)
    result(162) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array1(2)%a3
    result(163) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array1(2)%a4
    result(164) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array1(2)%a5

    result(165) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array1(3)%a1
    result(166) = ICHAR( tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array1(3)%a2)
    result(167) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array1(3)%a3
    result(168) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array1(3)%a4
    result(169) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array1(3)%a5

    result(170) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array2(1)%a1
    result(171) = ICHAR( tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array2(1)%a2)
    result(172) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array2(1)%a3
    result(173) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array2(1)%a4
    result(174) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array2(1)%a5

    result(175) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array2(2)%a1
    result(176) = ICHAR( tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array2(2)%a2)
    result(177) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array2(2)%a3
    result(178) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array2(2)%a4
    result(179) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array2(2)%a5

    result(180) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array2(3)%a1
    result(181) = ICHAR( tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array2(3)%a2)
    result(182) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array2(3)%a3
    result(183) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array2(3)%a4
    result(184) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array2(3)%a5

    result(185) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array4(1)%a1
    result(186) = ICHAR( tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array4(1)%a2)
    result(187) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array4(1)%a3
    result(188) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array4(1)%a4
    result(189) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array4(1)%a5

    result(190) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array4(2)%a1
    result(191) = ICHAR( tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array4(2)%a2)
    result(192) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array4(2)%a3
    result(193) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array4(2)%a4
    result(194) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array4(2)%a5

    result(195) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array4(3)%a1
    result(196) = ICHAR( tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array4(3)%a2)
    result(197) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array4(3)%a3
    result(198) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array4(3)%a4
    result(199) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array4(3)%a5

    result(200) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array5(1)%a1
    result(201) = ICHAR( tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array5(1)%a2)
    result(202) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array5(1)%a3
    result(203) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array5(1)%a4
    result(204) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array5(1)%a5

    result(205) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array5(2)%a1
    result(206) = ICHAR( tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array5(2)%a2)
    result(207) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array5(2)%a3
    result(208) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array5(2)%a4
    result(209) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array5(2)%a5

    result(210) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array5(3)%a1
    result(211) = ICHAR( tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array5(3)%a2)
    result(212) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array5(3)%a3
    result(213) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array5(3)%a4
    result(214) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array5(3)%a5

    result(215) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array6(1)%a1
    result(216) = ICHAR( tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array6(1)%a2)
    result(217) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array6(1)%a3
    result(218) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array6(1)%a4
    result(219) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array6(1)%a5

    result(220) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array6(2)%a1
    result(221) = ICHAR( tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array6(2)%a2)
    result(222) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array6(2)%a3
    result(223) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array6(2)%a4
    result(224) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array6(2)%a5

    result(225) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array6(3)%a1
    result(226) = ICHAR( tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array6(3)%a2)
    result(227) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array6(3)%a3
    result(228) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array6(3)%a4
    result(229) = tdc_inst_ptr1%tdb_inst_mbr1%tda_mbr_array6(3)%a5
    result(230) = tdc_inst_ptr1%tdb_inst_mbr1%b2
    result(231) = tdc_inst_ptr1%b2

    allocate(tdb_alloc_array1(2))

    result(232) = tdb_alloc_array1(1)%b1
    result(233) = tdb_alloc_array1(1)%tda_mbr_inst1%a1
    result(234) = ICHAR(tdb_alloc_array1(1)%tda_mbr_inst1%a2)
    result(235) = tdb_alloc_array1(1)%tda_mbr_inst1%a3
    result(236) = tdb_alloc_array1(1)%tda_mbr_inst1%a4
    result(237) = tdb_alloc_array1(1)%tda_mbr_inst1%a5

    result(238) = tdb_alloc_array1(1)%tda_mbr_inst2%a1
    result(239) = ICHAR(tdb_alloc_array1(1)%tda_mbr_inst2%a2)
    result(240) = tdb_alloc_array1(1)%tda_mbr_inst2%a3
    result(241) = tdb_alloc_array1(1)%tda_mbr_inst2%a4
    result(242) = tdb_alloc_array1(1)%tda_mbr_inst2%a5

    result(243) = tdb_alloc_array1(1)%tda_mbr_array1(1)%a1
    result(244) = ICHAR( tdb_alloc_array1(1)%tda_mbr_array1(1)%a2)
    result(245) = tdb_alloc_array1(1)%tda_mbr_array1(1)%a3
    result(246) = tdb_alloc_array1(1)%tda_mbr_array1(1)%a4
    result(247) = tdb_alloc_array1(1)%tda_mbr_array1(1)%a5

    result(248) = tdb_alloc_array1(1)%tda_mbr_array1(2)%a1
    result(249) = ICHAR( tdb_alloc_array1(1)%tda_mbr_array1(2)%a2)
    result(250) = tdb_alloc_array1(1)%tda_mbr_array1(2)%a3
    result(251) = tdb_alloc_array1(1)%tda_mbr_array1(2)%a4
    result(252) = tdb_alloc_array1(1)%tda_mbr_array1(2)%a5

    result(253) = tdb_alloc_array1(1)%tda_mbr_array1(3)%a1
    result(254) = ICHAR( tdb_alloc_array1(1)%tda_mbr_array1(3)%a2)
    result(255) = tdb_alloc_array1(1)%tda_mbr_array1(3)%a3
    result(256) = tdb_alloc_array1(1)%tda_mbr_array1(3)%a4
    result(257) = tdb_alloc_array1(1)%tda_mbr_array1(3)%a5

    result(258) = tdb_alloc_array1(1)%tda_mbr_array2(1)%a1
    result(259) = ICHAR( tdb_alloc_array1(1)%tda_mbr_array2(1)%a2)
    result(260) = tdb_alloc_array1(1)%tda_mbr_array2(1)%a3
    result(261) = tdb_alloc_array1(1)%tda_mbr_array2(1)%a4
    result(262) = tdb_alloc_array1(1)%tda_mbr_array2(1)%a5

    result(263) = tdb_alloc_array1(1)%tda_mbr_array2(2)%a1
    result(264) = ICHAR( tdb_alloc_array1(1)%tda_mbr_array2(2)%a2)
    result(265) = tdb_alloc_array1(1)%tda_mbr_array2(2)%a3
    result(266) = tdb_alloc_array1(1)%tda_mbr_array2(2)%a4
    result(267) = tdb_alloc_array1(1)%tda_mbr_array2(2)%a5

    result(268) = tdb_alloc_array1(1)%tda_mbr_array2(3)%a1
    result(269) = ICHAR( tdb_alloc_array1(1)%tda_mbr_array2(3)%a2)
    result(270) = tdb_alloc_array1(1)%tda_mbr_array2(3)%a3
    result(271) = tdb_alloc_array1(1)%tda_mbr_array2(3)%a4
    result(272) = tdb_alloc_array1(1)%tda_mbr_array2(3)%a5

    result(273) = tdb_alloc_array1(1)%tda_mbr_array4(1)%a1
    result(274) = ICHAR( tdb_alloc_array1(1)%tda_mbr_array4(1)%a2)
    result(275) = tdb_alloc_array1(1)%tda_mbr_array4(1)%a3
    result(276) = tdb_alloc_array1(1)%tda_mbr_array4(1)%a4
    result(277) = tdb_alloc_array1(1)%tda_mbr_array4(1)%a5

    result(278) = tdb_alloc_array1(1)%tda_mbr_array4(2)%a1
    result(279) = ICHAR( tdb_alloc_array1(1)%tda_mbr_array4(2)%a2)
    result(280) = tdb_alloc_array1(1)%tda_mbr_array4(2)%a3
    result(281) = tdb_alloc_array1(1)%tda_mbr_array4(2)%a4
    result(282) = tdb_alloc_array1(1)%tda_mbr_array4(2)%a5

    result(283) = tdb_alloc_array1(1)%tda_mbr_array4(3)%a1
    result(284) = ICHAR( tdb_alloc_array1(1)%tda_mbr_array4(3)%a2)
    result(285) = tdb_alloc_array1(1)%tda_mbr_array4(3)%a3
    result(286) = tdb_alloc_array1(1)%tda_mbr_array4(3)%a4
    result(287) = tdb_alloc_array1(1)%tda_mbr_array4(3)%a5

    result(288) = tdb_alloc_array1(1)%tda_mbr_array5(1)%a1
    result(289) = ICHAR( tdb_alloc_array1(1)%tda_mbr_array5(1)%a2)
    result(290) = tdb_alloc_array1(1)%tda_mbr_array5(1)%a3
    result(291) = tdb_alloc_array1(1)%tda_mbr_array5(1)%a4
    result(292) = tdb_alloc_array1(1)%tda_mbr_array5(1)%a5

    result(293) = tdb_alloc_array1(1)%tda_mbr_array5(2)%a1
    result(294) = ICHAR( tdb_alloc_array1(1)%tda_mbr_array5(2)%a2)
    result(295) = tdb_alloc_array1(1)%tda_mbr_array5(2)%a3
    result(296) = tdb_alloc_array1(1)%tda_mbr_array5(2)%a4
    result(297) = tdb_alloc_array1(1)%tda_mbr_array5(2)%a5

    result(298) = tdb_alloc_array1(1)%tda_mbr_array5(3)%a1
    result(299) = ICHAR( tdb_alloc_array1(1)%tda_mbr_array5(3)%a2)
    result(300) = tdb_alloc_array1(1)%tda_mbr_array5(3)%a3
    result(301) = tdb_alloc_array1(1)%tda_mbr_array5(3)%a4
    result(302) = tdb_alloc_array1(1)%tda_mbr_array5(3)%a5

    result(303) = tdb_alloc_array1(1)%tda_mbr_array6(1)%a1
    result(304) = ICHAR( tdb_alloc_array1(1)%tda_mbr_array6(1)%a2)
    result(305) = tdb_alloc_array1(1)%tda_mbr_array6(1)%a3
    result(306) = tdb_alloc_array1(1)%tda_mbr_array6(1)%a4
    result(307) = tdb_alloc_array1(1)%tda_mbr_array6(1)%a5

    result(308) = tdb_alloc_array1(1)%tda_mbr_array6(2)%a1
    result(309) = ICHAR( tdb_alloc_array1(1)%tda_mbr_array6(2)%a2)
    result(310) = tdb_alloc_array1(1)%tda_mbr_array6(2)%a3
    result(311) = tdb_alloc_array1(1)%tda_mbr_array6(2)%a4
    result(312) = tdb_alloc_array1(1)%tda_mbr_array6(2)%a5

    result(313) = tdb_alloc_array1(1)%tda_mbr_array6(3)%a1
    result(314) = ICHAR( tdb_alloc_array1(1)%tda_mbr_array6(3)%a2)
    result(315) = tdb_alloc_array1(1)%tda_mbr_array6(3)%a3
    result(316) = tdb_alloc_array1(1)%tda_mbr_array6(3)%a4
    result(317) = tdb_alloc_array1(1)%tda_mbr_array6(3)%a5
    result(318) = tdb_alloc_array1(1)%b2

    result(319) = tdb_alloc_array1(2)%b1
    result(320) = tdb_alloc_array1(2)%tda_mbr_inst1%a1
    result(321) = ICHAR(tdb_alloc_array1(1)%tda_mbr_inst1%a2)
    result(322) = tdb_alloc_array1(2)%tda_mbr_inst1%a3
    result(323) = tdb_alloc_array1(2)%tda_mbr_inst1%a4
    result(324) = tdb_alloc_array1(2)%tda_mbr_inst1%a5

    result(325) = tdb_alloc_array1(2)%tda_mbr_inst2%a1
    result(326) = ICHAR(tdb_alloc_array1(1)%tda_mbr_inst2%a2)
    result(327) = tdb_alloc_array1(2)%tda_mbr_inst2%a3
    result(328) = tdb_alloc_array1(2)%tda_mbr_inst2%a4
    result(329) = tdb_alloc_array1(2)%tda_mbr_inst2%a5

    result(330) = tdb_alloc_array1(2)%tda_mbr_array1(1)%a1
    result(331) = ICHAR( tdb_alloc_array1(2)%tda_mbr_array1(1)%a2)
    result(332) = tdb_alloc_array1(2)%tda_mbr_array1(1)%a3
    result(333) = tdb_alloc_array1(2)%tda_mbr_array1(1)%a4
    result(334) = tdb_alloc_array1(2)%tda_mbr_array1(1)%a5

    result(335) = tdb_alloc_array1(2)%tda_mbr_array1(2)%a1
    result(336) = ICHAR( tdb_alloc_array1(2)%tda_mbr_array1(2)%a2)
    result(337) = tdb_alloc_array1(2)%tda_mbr_array1(2)%a3
    result(338) = tdb_alloc_array1(2)%tda_mbr_array1(2)%a4
    result(339) = tdb_alloc_array1(2)%tda_mbr_array1(2)%a5

    result(340) = tdb_alloc_array1(2)%tda_mbr_array1(3)%a1
    result(341) = ICHAR( tdb_alloc_array1(2)%tda_mbr_array1(3)%a2)
    result(342) = tdb_alloc_array1(2)%tda_mbr_array1(3)%a3
    result(343) = tdb_alloc_array1(2)%tda_mbr_array1(3)%a4
    result(344) = tdb_alloc_array1(2)%tda_mbr_array1(3)%a5

    result(345) = tdb_alloc_array1(2)%tda_mbr_array2(1)%a1
    result(346) = ICHAR( tdb_alloc_array1(2)%tda_mbr_array2(1)%a2)
    result(347) = tdb_alloc_array1(2)%tda_mbr_array2(1)%a3
    result(348) = tdb_alloc_array1(2)%tda_mbr_array2(1)%a4
    result(349) = tdb_alloc_array1(2)%tda_mbr_array2(1)%a5

    result(350) = tdb_alloc_array1(2)%tda_mbr_array2(2)%a1
    result(351) = ICHAR( tdb_alloc_array1(2)%tda_mbr_array2(2)%a2)
    result(352) = tdb_alloc_array1(2)%tda_mbr_array2(2)%a3
    result(353) = tdb_alloc_array1(2)%tda_mbr_array2(2)%a4
    result(354) = tdb_alloc_array1(2)%tda_mbr_array2(2)%a5

    result(355) = tdb_alloc_array1(2)%tda_mbr_array2(3)%a1
    result(356) = ICHAR( tdb_alloc_array1(2)%tda_mbr_array2(3)%a2)
    result(357) = tdb_alloc_array1(2)%tda_mbr_array2(3)%a3
    result(358) = tdb_alloc_array1(2)%tda_mbr_array2(3)%a4
    result(359) = tdb_alloc_array1(2)%tda_mbr_array2(3)%a5

    result(360) = tdb_alloc_array1(2)%tda_mbr_array4(1)%a1
    result(361) = ICHAR( tdb_alloc_array1(2)%tda_mbr_array4(1)%a2)
    result(362) = tdb_alloc_array1(2)%tda_mbr_array4(1)%a3
    result(363) = tdb_alloc_array1(2)%tda_mbr_array4(1)%a4
    result(364) = tdb_alloc_array1(2)%tda_mbr_array4(1)%a5

    result(365) = tdb_alloc_array1(2)%tda_mbr_array4(2)%a1
    result(366) = ICHAR( tdb_alloc_array1(2)%tda_mbr_array4(2)%a2)
    result(367) = tdb_alloc_array1(2)%tda_mbr_array4(2)%a3
    result(368) = tdb_alloc_array1(2)%tda_mbr_array4(2)%a4
    result(369) = tdb_alloc_array1(2)%tda_mbr_array4(2)%a5

    result(370) = tdb_alloc_array1(2)%tda_mbr_array4(3)%a1
    result(371) = ICHAR( tdb_alloc_array1(2)%tda_mbr_array4(3)%a2)
    result(372) = tdb_alloc_array1(2)%tda_mbr_array4(3)%a3
    result(373) = tdb_alloc_array1(2)%tda_mbr_array4(3)%a4
    result(374) = tdb_alloc_array1(2)%tda_mbr_array4(3)%a5

    result(375) = tdb_alloc_array1(2)%tda_mbr_array5(1)%a1
    result(376) = ICHAR( tdb_alloc_array1(2)%tda_mbr_array5(1)%a2)
    result(377) = tdb_alloc_array1(2)%tda_mbr_array5(1)%a3
    result(378) = tdb_alloc_array1(2)%tda_mbr_array5(1)%a4
    result(379) = tdb_alloc_array1(2)%tda_mbr_array5(1)%a5

    result(380) = tdb_alloc_array1(2)%tda_mbr_array5(2)%a1
    result(381) = ICHAR( tdb_alloc_array1(2)%tda_mbr_array5(2)%a2)
    result(382) = tdb_alloc_array1(2)%tda_mbr_array5(2)%a3
    result(383) = tdb_alloc_array1(2)%tda_mbr_array5(2)%a4
    result(384) = tdb_alloc_array1(2)%tda_mbr_array5(2)%a5

    result(385) = tdb_alloc_array1(2)%tda_mbr_array5(3)%a1
    result(386) = ICHAR( tdb_alloc_array1(2)%tda_mbr_array5(3)%a2)
    result(387) = tdb_alloc_array1(2)%tda_mbr_array5(3)%a3
    result(388) = tdb_alloc_array1(2)%tda_mbr_array5(3)%a4
    result(389) = tdb_alloc_array1(2)%tda_mbr_array5(3)%a5

    result(390) = tdb_alloc_array1(2)%tda_mbr_array6(1)%a1
    result(391) = ICHAR( tdb_alloc_array1(2)%tda_mbr_array6(1)%a2)
    result(392) = tdb_alloc_array1(2)%tda_mbr_array6(1)%a3
    result(393) = tdb_alloc_array1(2)%tda_mbr_array6(1)%a4
    result(394) = tdb_alloc_array1(2)%tda_mbr_array6(1)%a5

    result(395) = tdb_alloc_array1(2)%tda_mbr_array6(2)%a1
    result(396) = ICHAR( tdb_alloc_array1(2)%tda_mbr_array6(2)%a2)
    result(397) = tdb_alloc_array1(2)%tda_mbr_array6(2)%a3
    result(398) = tdb_alloc_array1(2)%tda_mbr_array6(2)%a4
    result(399) = tdb_alloc_array1(2)%tda_mbr_array6(2)%a5

    result(400) = tdb_alloc_array1(2)%tda_mbr_array6(3)%a1
    result(401) = ICHAR( tdb_alloc_array1(2)%tda_mbr_array6(3)%a2)
    result(402) = tdb_alloc_array1(2)%tda_mbr_array6(3)%a3
    result(403) = tdb_alloc_array1(2)%tda_mbr_array6(3)%a4
    result(404) = tdb_alloc_array1(2)%tda_mbr_array6(3)%a5
    result(405) = tdb_alloc_array1(2)%b2

    result(406) = tdb_inst%b1
    result(407) = tdb_inst%tda_mbr_inst1%a1
    result(408) = ICHAR(tdb_inst%tda_mbr_inst1%a2)
    result(409) = tdb_inst%tda_mbr_inst1%a3
    result(410) = tdb_inst%tda_mbr_inst1%a4
    result(411) = tdb_inst%tda_mbr_inst1%a5

    result(412) = tdb_inst%tda_mbr_inst2%a1
    result(413) = ICHAR(tdb_inst%tda_mbr_inst2%a2)
    result(414) = tdb_inst%tda_mbr_inst2%a3
    result(415) = tdb_inst%tda_mbr_inst2%a4
    result(416) = tdb_inst%tda_mbr_inst2%a5

    result(417) = tdb_inst%tda_mbr_array1(1)%a1
    result(418) = ICHAR( tdb_inst%tda_mbr_array1(1)%a2)
    result(419) = tdb_inst%tda_mbr_array1(1)%a3
    result(420) = tdb_inst%tda_mbr_array1(1)%a4
    result(421) = tdb_inst%tda_mbr_array1(1)%a5

    result(422) = tdb_inst%tda_mbr_array1(2)%a1
    result(423) = ICHAR( tdb_inst%tda_mbr_array1(2)%a2)
    result(424) = tdb_inst%tda_mbr_array1(2)%a3
    result(425) = tdb_inst%tda_mbr_array1(2)%a4
    result(426) = tdb_inst%tda_mbr_array1(2)%a5

    result(427) = tdb_inst%tda_mbr_array1(3)%a1
    result(428) = ICHAR( tdb_inst%tda_mbr_array1(3)%a2)
    result(429) = tdb_inst%tda_mbr_array1(3)%a3
    result(430) = tdb_inst%tda_mbr_array1(3)%a4
    result(431) = tdb_inst%tda_mbr_array1(3)%a5

    result(432) = tdb_inst%tda_mbr_array2(1)%a1
    result(433) = ICHAR( tdb_inst%tda_mbr_array2(1)%a2)
    result(434) = tdb_inst%tda_mbr_array2(1)%a3
    result(435) = tdb_inst%tda_mbr_array2(1)%a4
    result(436) = tdb_inst%tda_mbr_array2(1)%a5

    result(437) = tdb_inst%tda_mbr_array2(2)%a1
    result(438) = ICHAR( tdb_inst%tda_mbr_array2(2)%a2)
    result(439) = tdb_inst%tda_mbr_array2(2)%a3
    result(440) = tdb_inst%tda_mbr_array2(2)%a4
    result(441) = tdb_inst%tda_mbr_array2(2)%a5

    result(442) = tdb_inst%tda_mbr_array2(3)%a1
    result(443) = ICHAR( tdb_inst%tda_mbr_array2(3)%a2)
    result(444) = tdb_inst%tda_mbr_array2(3)%a3
    result(445) = tdb_inst%tda_mbr_array2(3)%a4
    result(446) = tdb_inst%tda_mbr_array2(3)%a5

    result(447) = tdb_inst%tda_mbr_array4(1)%a1
    result(448) = ICHAR( tdb_inst%tda_mbr_array4(1)%a2)
    result(449) = tdb_inst%tda_mbr_array4(1)%a3
    result(450) = tdb_inst%tda_mbr_array4(1)%a4
    result(451) = tdb_inst%tda_mbr_array4(1)%a5

    result(452) = tdb_inst%tda_mbr_array4(2)%a1
    result(453) = ICHAR( tdb_inst%tda_mbr_array4(2)%a2)
    result(454) = tdb_inst%tda_mbr_array4(2)%a3
    result(455) = tdb_inst%tda_mbr_array4(2)%a4
    result(456) = tdb_inst%tda_mbr_array4(2)%a5

    result(457) = tdb_inst%tda_mbr_array4(3)%a1
    result(458) = ICHAR( tdb_inst%tda_mbr_array4(3)%a2)
    result(459) = tdb_inst%tda_mbr_array4(3)%a3
    result(460) = tdb_inst%tda_mbr_array4(3)%a4
    result(461) = tdb_inst%tda_mbr_array4(3)%a5

    result(462) = tdb_inst%tda_mbr_array5(1)%a1
    result(463) = ICHAR( tdb_inst%tda_mbr_array5(1)%a2)
    result(464) = tdb_inst%tda_mbr_array5(1)%a3
    result(465) = tdb_inst%tda_mbr_array5(1)%a4
    result(466) = tdb_inst%tda_mbr_array5(1)%a5

    result(467) = tdb_inst%tda_mbr_array5(2)%a1
    result(468) = ICHAR( tdb_inst%tda_mbr_array5(2)%a2)
    result(469) = tdb_inst%tda_mbr_array5(2)%a3
    result(470) = tdb_inst%tda_mbr_array5(2)%a4
    result(471) = tdb_inst%tda_mbr_array5(2)%a5

    result(472) = tdb_inst%tda_mbr_array5(3)%a1
    result(473) = ICHAR( tdb_inst%tda_mbr_array5(3)%a2)
    result(474) = tdb_inst%tda_mbr_array5(3)%a3
    result(475) = tdb_inst%tda_mbr_array5(3)%a4
    result(476) = tdb_inst%tda_mbr_array5(3)%a5

    result(477) = tdb_inst%tda_mbr_array6(1)%a1
    result(478) = ICHAR( tdb_inst%tda_mbr_array6(1)%a2)
    result(479) = tdb_inst%tda_mbr_array6(1)%a3
    result(480) = tdb_inst%tda_mbr_array6(1)%a4
    result(481) = tdb_inst%tda_mbr_array6(1)%a5

    result(482) = tdb_inst%tda_mbr_array6(2)%a1
    result(483) = ICHAR( tdb_inst%tda_mbr_array6(2)%a2)
    result(484) = tdb_inst%tda_mbr_array6(2)%a3
    result(485) = tdb_inst%tda_mbr_array6(2)%a4
    result(486) = tdb_inst%tda_mbr_array6(2)%a5

    result(487) = tdb_inst%tda_mbr_array6(3)%a1
    result(488) = ICHAR( tdb_inst%tda_mbr_array6(3)%a2)
    result(489) = tdb_inst%tda_mbr_array6(3)%a3
    result(490) = tdb_inst%tda_mbr_array6(3)%a4
    result(491) = tdb_inst%tda_mbr_array6(3)%a5
    result(492) = tdb_inst%b2

    result(493) = tdc_inst%b1
    result(494) = tdc_inst%tda_inst_mbr1%a1
    result(495) = ICHAR( tdc_inst%tda_inst_mbr1%a2)
    result(496) = tdc_inst%tda_inst_mbr1%a3
    result(497) = tdc_inst%tda_inst_mbr1%a4
    result(498) = tdc_inst%tda_inst_mbr1%a5

    result(499) = tdc_inst%tda_inst_mbr2%a1
    result(500) = ICHAR( tdc_inst%tda_inst_mbr2%a2)
    result(501) = tdc_inst%tda_inst_mbr2%a3
    result(502) = tdc_inst%tda_inst_mbr2%a4
    result(503) = tdc_inst%tda_inst_mbr2%a5

    result(504) = tdc_inst%tda_inst_mbr3%a1
    result(505) = ICHAR( tdc_inst%tda_inst_mbr3%a2)
    result(506) = tdc_inst%tda_inst_mbr3%a3
    result(507) = tdc_inst%tda_inst_mbr3%a4
    result(508) = tdc_inst%tda_inst_mbr3%a5

    result(509) = tdc_inst%tdb_inst_mbr1%b1
    result(510) = tdc_inst%tdb_inst_mbr1%tda_mbr_inst1%a1
    result(511) = ICHAR(tdc_inst%tdb_inst_mbr1%tda_mbr_inst1%a2)
    result(512) = tdc_inst%tdb_inst_mbr1%tda_mbr_inst1%a3
    result(513) = tdc_inst%tdb_inst_mbr1%tda_mbr_inst1%a4
    result(514) = tdc_inst%tdb_inst_mbr1%tda_mbr_inst1%a5

    result(515) = tdc_inst%tdb_inst_mbr1%tda_mbr_inst2%a1
    result(516) = ICHAR(tdc_inst%tdb_inst_mbr1%tda_mbr_inst2%a2)
    result(517) = tdc_inst%tdb_inst_mbr1%tda_mbr_inst2%a3
    result(518) = tdc_inst%tdb_inst_mbr1%tda_mbr_inst2%a4
    result(519) = tdc_inst%tdb_inst_mbr1%tda_mbr_inst2%a5

    result(520) = tdc_inst%tdb_inst_mbr1%tda_mbr_array1(1)%a1
    result(521) = ICHAR( tdc_inst%tdb_inst_mbr1%tda_mbr_array1(1)%a2)
    result(522) = tdc_inst%tdb_inst_mbr1%tda_mbr_array1(1)%a3
    result(523) = tdc_inst%tdb_inst_mbr1%tda_mbr_array1(1)%a4
    result(524) = tdc_inst%tdb_inst_mbr1%tda_mbr_array1(1)%a5

    result(525) = tdc_inst%tdb_inst_mbr1%tda_mbr_array1(2)%a1
    result(526) = ICHAR( tdc_inst%tdb_inst_mbr1%tda_mbr_array1(2)%a2)
    result(527) = tdc_inst%tdb_inst_mbr1%tda_mbr_array1(2)%a3
    result(528) = tdc_inst%tdb_inst_mbr1%tda_mbr_array1(2)%a4
    result(529) = tdc_inst%tdb_inst_mbr1%tda_mbr_array1(2)%a5

    result(530) = tdc_inst%tdb_inst_mbr1%tda_mbr_array1(3)%a1
    result(531) = ICHAR( tdc_inst%tdb_inst_mbr1%tda_mbr_array1(3)%a2)
    result(532) = tdc_inst%tdb_inst_mbr1%tda_mbr_array1(3)%a3
    result(533) = tdc_inst%tdb_inst_mbr1%tda_mbr_array1(3)%a4
    result(534) = tdc_inst%tdb_inst_mbr1%tda_mbr_array1(3)%a5

    result(535) = tdc_inst%tdb_inst_mbr1%tda_mbr_array2(1)%a1
    result(536) = ICHAR( tdc_inst%tdb_inst_mbr1%tda_mbr_array2(1)%a2)
    result(537) = tdc_inst%tdb_inst_mbr1%tda_mbr_array2(1)%a3
    result(538) = tdc_inst%tdb_inst_mbr1%tda_mbr_array2(1)%a4
    result(539) = tdc_inst%tdb_inst_mbr1%tda_mbr_array2(1)%a5

    result(540) = tdc_inst%tdb_inst_mbr1%tda_mbr_array2(2)%a1
    result(541) = ICHAR( tdc_inst%tdb_inst_mbr1%tda_mbr_array2(2)%a2)
    result(542) = tdc_inst%tdb_inst_mbr1%tda_mbr_array2(2)%a3
    result(543) = tdc_inst%tdb_inst_mbr1%tda_mbr_array2(2)%a4
    result(544) = tdc_inst%tdb_inst_mbr1%tda_mbr_array2(2)%a5

    result(545) = tdc_inst%tdb_inst_mbr1%tda_mbr_array2(3)%a1
    result(546) = ICHAR( tdc_inst%tdb_inst_mbr1%tda_mbr_array2(3)%a2)
    result(547) = tdc_inst%tdb_inst_mbr1%tda_mbr_array2(3)%a3
    result(548) = tdc_inst%tdb_inst_mbr1%tda_mbr_array2(3)%a4
    result(549) = tdc_inst%tdb_inst_mbr1%tda_mbr_array2(3)%a5

    result(550) = tdc_inst%tdb_inst_mbr1%tda_mbr_array4(1)%a1
    result(551) = ICHAR( tdc_inst%tdb_inst_mbr1%tda_mbr_array4(1)%a2)
    result(552) = tdc_inst%tdb_inst_mbr1%tda_mbr_array4(1)%a3
    result(553) = tdc_inst%tdb_inst_mbr1%tda_mbr_array4(1)%a4
    result(554) = tdc_inst%tdb_inst_mbr1%tda_mbr_array4(1)%a5

    result(555) = tdc_inst%tdb_inst_mbr1%tda_mbr_array4(2)%a1
    result(556) = ICHAR( tdc_inst%tdb_inst_mbr1%tda_mbr_array4(2)%a2)
    result(557) = tdc_inst%tdb_inst_mbr1%tda_mbr_array4(2)%a3
    result(558) = tdc_inst%tdb_inst_mbr1%tda_mbr_array4(2)%a4
    result(559) = tdc_inst%tdb_inst_mbr1%tda_mbr_array4(2)%a5

    result(560) = tdc_inst%tdb_inst_mbr1%tda_mbr_array4(3)%a1
    result(561) = ICHAR( tdc_inst%tdb_inst_mbr1%tda_mbr_array4(3)%a2)
    result(562) = tdc_inst%tdb_inst_mbr1%tda_mbr_array4(3)%a3
    result(563) = tdc_inst%tdb_inst_mbr1%tda_mbr_array4(3)%a4
    result(564) = tdc_inst%tdb_inst_mbr1%tda_mbr_array4(3)%a5

    result(565) = tdc_inst%tdb_inst_mbr1%tda_mbr_array5(1)%a1
    result(566) = ICHAR( tdc_inst%tdb_inst_mbr1%tda_mbr_array5(1)%a2)
    result(567) = tdc_inst%tdb_inst_mbr1%tda_mbr_array5(1)%a3
    result(568) = tdc_inst%tdb_inst_mbr1%tda_mbr_array5(1)%a4
    result(569) = tdc_inst%tdb_inst_mbr1%tda_mbr_array5(1)%a5
    result(570) = tdc_inst%tdb_inst_mbr1%tda_mbr_array5(2)%a1

    result(571) = ICHAR( tdc_inst%tdb_inst_mbr1%tda_mbr_array5(2)%a2)
    result(572) = tdc_inst%tdb_inst_mbr1%tda_mbr_array5(2)%a3
    result(573) = tdc_inst%tdb_inst_mbr1%tda_mbr_array5(2)%a4
    result(574) = tdc_inst%tdb_inst_mbr1%tda_mbr_array5(2)%a5
    result(575) = tdc_inst%tdb_inst_mbr1%tda_mbr_array5(3)%a1

    result(576) = ICHAR( tdc_inst%tdb_inst_mbr1%tda_mbr_array5(3)%a2)
    result(577) = tdc_inst%tdb_inst_mbr1%tda_mbr_array5(3)%a3
    result(578) = tdc_inst%tdb_inst_mbr1%tda_mbr_array5(3)%a4
    result(579) = tdc_inst%tdb_inst_mbr1%tda_mbr_array5(3)%a5
    result(580) = tdc_inst%tdb_inst_mbr1%tda_mbr_array6(1)%a1

    result(581) = ICHAR( tdc_inst%tdb_inst_mbr1%tda_mbr_array6(1)%a2)
    result(582) = tdc_inst%tdb_inst_mbr1%tda_mbr_array6(1)%a3
    result(583) = tdc_inst%tdb_inst_mbr1%tda_mbr_array6(1)%a4
    result(584) = tdc_inst%tdb_inst_mbr1%tda_mbr_array6(1)%a5
    result(585) = tdc_inst%tdb_inst_mbr1%tda_mbr_array6(2)%a1

    result(586) = ICHAR( tdc_inst%tdb_inst_mbr1%tda_mbr_array6(2)%a2)
    result(587) = tdc_inst%tdb_inst_mbr1%tda_mbr_array6(2)%a3
    result(588) = tdc_inst%tdb_inst_mbr1%tda_mbr_array6(2)%a4
    result(589) = tdc_inst%tdb_inst_mbr1%tda_mbr_array6(2)%a5
    result(590) = tdc_inst%tdb_inst_mbr1%tda_mbr_array6(3)%a1

    result(591) = ICHAR( tdc_inst%tdb_inst_mbr1%tda_mbr_array6(3)%a2)
    result(592) = tdc_inst%tdb_inst_mbr1%tda_mbr_array6(3)%a3
    result(593) = tdc_inst%tdb_inst_mbr1%tda_mbr_array6(3)%a4
    result(594) = tdc_inst%tdb_inst_mbr1%tda_mbr_array6(3)%a5
    result(595) = tdc_inst%tdb_inst_mbr1%b2
    result(596) = tdc_inst%b2

    result(597) = tdd_inst%tdd_int1
    result(598:602) = tdd_inst%tdd_intArray1
    result(603:607) = tdd_inst%tdd_intArray2
    result(608:612) = tdd_inst%tdd_intArray3
    result(613:615) = tdd_inst%tdd_intArray4
    result(616:618) = tdd_inst%tdd_intArray5
    result(619:621) = tdd_inst%tdd_intArray6
    result(622:626) = tdd_inst%tdd_intArray7
    result(627:631) = tdd_inst%tdd_intArray8
    result(632:636) = tdd_inst%tdd_intArray9

    allocate(tdd_ptr1);
    result(637) = tdd_ptr1%tdd_int1
    result(638:642) = tdd_ptr1%tdd_intArray1
    result(643:647) = tdd_ptr1%tdd_intArray2
    result(648:652) = tdd_ptr1%tdd_intArray3
    result(653:655) = tdd_ptr1%tdd_intArray4
    result(656:658) = tdd_ptr1%tdd_intArray5
    result(659:661) = tdd_ptr1%tdd_intArray6
    result(662:666) = tdd_ptr1%tdd_intArray7
    result(667:671) = tdd_ptr1%tdd_intArray8
    result(672:676) = tdd_ptr1%tdd_intArray9

    call check(result, expect, N)
  end
