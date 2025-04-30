!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!* Tests for runtime library MATMUL routines

program p
  
  parameter(NbrTests=1056)
  parameter(o_extent=2)
  parameter(n_extent=6)
  parameter(m_extent=4)
  parameter(k_extent=8)
  
  real*8, target, dimension(n_extent,m_extent) :: arr1
  real*8, target, dimension(m_extent,k_extent) :: arr2
  real*8, target, dimension(n_extent,k_extent) :: arr3
  
  real*8, pointer, dimension(:,:) :: arr1_ptr
  real*8, pointer, dimension(:,:) :: arr2_ptr
  real*8, pointer, dimension(:,:) :: arr3_ptr
  
  type dtype1
    real*8, pointer, dimension(:,:) :: arr1_ptr
  end type

  type dtype2
    character :: c
    real*8, dimension(m_extent,k_extent) :: arr2
  end type

  type dtype3a
    character :: c
    real*8, dimension(n_extent,k_extent) :: arr3
  end type

  type dtype3b
    character :: c
    real*8, pointer, dimension(:,:) :: arr3_ptr
  end type

  type dtype4
    character :: c
    real*8 :: r
  end type

  type dtype5
    integer :: i
    real*8 :: r
    character :: c
  end type

  type (dtype1) :: d1_inst
  type (dtype2) :: d2_inst
  type (dtype3a) :: d3a_inst
  type (dtype3b) :: d3b_inst
  type (dtype4) :: d4_inst(n_extent,k_extent)
  type (dtype5) :: d5_inst(m_extent,k_extent)

  integer :: one, two, four, mminus1
  
  REAL*8 :: expect(NbrTests) 
  REAL*8 :: results(NbrTests)
  
  integer:: i,j
  
  data arr1 /0,1,2,3,4,5,6,7,8,9,10,11,			&
             12,13,14,15,16,17,18,19,20,22,22,23/
  data arr2 /0,1,2,3,4,5,6,7,8,9,10,11,			&
             12,13,14,15,16,17,18,19,20,21,22,23,		&
             24,25,26,27,28,29,30,31/
  data arr3 /0,1,2,3,4,5,6,7,8,9,10,11, 			&
             12,13,14,15,16,17,18,19,20,21,22,23,		&
             24,25,26,27,28,29,30,31,32,33,34,35,		&
             36,37,38,39,40,41,42,43,44,45,46,47/
  
  data expect /  &
  ! test 1-48
      84.0, 90.0, 96.0, 105.0, 108.0, 114.0, 228.0, 250.0, 272.0, &
      301.0, 316.0, 338.0, 372.0, 410.0, 448.0, 497.0, 524.0, 562.0, &
      516.0, 570.0, 624.0, 693.0, 732.0, 786.0, 660.0, 730.0, 800.0, &
      889.0, 940.0, 1010.0, 804.0, 890.0, 976.0, 1085.0, 1148.0, 1234.0, &
      948.0, 1050.0, 1152.0, 1281.0, 1356.0, 1458.0, 1092.0, 1210.0, 1328.0, &
      1477.0, 1564.0, 1682.0, &
  ! test 49-96
      0.0, 90.0, 96.0, 105.0, 108.0, 114.0, 0.0, 250.0, 272.0, &
      301.0, 316.0, 338.0, 0.0, 410.0, 448.0, 497.0, 524.0, 562.0, &
      0.0, 570.0, 624.0, 693.0, 732.0, 786.0, 0.0, 730.0, 800.0, &
      889.0, 940.0, 1010.0, 0.0, 890.0, 976.0, 1085.0, 1148.0, 1234.0, &
      0.0, 1050.0, 1152.0, 1281.0, 1356.0, 1458.0, 0.0, 1210.0, 1328.0, &
      1477.0, 1564.0, 1682.0, &
  ! test 97-144
      84.0, 90.0, 96.0, 105.0, 108.0, 0.0, 228.0, 250.0, 272.0, &
      301.0, 316.0, 0.0, 372.0, 410.0, 448.0, 497.0, 524.0, 0.0, &
      516.0, 570.0, 624.0, 693.0, 732.0, 0.0, 660.0, 730.0, 800.0, &
      889.0, 940.0, 0.0, 804.0, 890.0, 976.0, 1085.0, 1148.0, 0.0, &
      948.0, 1050.0, 1152.0, 1281.0, 1356.0, 0.0, 1092.0, 1210.0, 1328.0, &
      1477.0, 1564.0, 0.0, &
  ! test 145-192
      84.0, 90.0, 96.0, 105.0, 108.0, 114.0, 228.0, 246.0, 264.0, &
      289.0, 300.0, 318.0, 372.0, 402.0, 432.0, 473.0, 492.0, 522.0, &
      516.0, 558.0, 600.0, 657.0, 684.0, 726.0, 660.0, 714.0, 768.0, &
      841.0, 876.0, 930.0, 804.0, 870.0, 936.0, 1025.0, 1068.0, 1134.0, &
      948.0, 1026.0, 1104.0, 1209.0, 1260.0, 1338.0, 1092.0, 1182.0, 1272.0, &
      1393.0, 1452.0, 1542.0, &
  ! test 193-240
      84.0, 90.0, 96.0, 105.0, 108.0, 114.0, 228.0, 246.0, 264.0, &
      289.0, 300.0, 318.0, 372.0, 402.0, 432.0, 473.0, 492.0, 522.0, &
      516.0, 558.0, 600.0, 657.0, 684.0, 726.0, 660.0, 714.0, 768.0, &
      841.0, 876.0, 930.0, 804.0, 870.0, 936.0, 1025.0, 1068.0, 1134.0, &
      948.0, 1026.0, 1104.0, 1209.0, 1260.0, 1338.0, 1092.0, 1182.0, 1272.0, &
      1393.0, 1452.0, 1542.0, &
  ! test 241-288
      30.0, 33.0, 36.0, 39.0, 42.0, 45.0, 102.0, 117.0, 132.0, &
      147.0, 162.0, 177.0, 174.0, 201.0, 228.0, 255.0, 282.0, 309.0, &
      246.0, 285.0, 324.0, 363.0, 402.0, 441.0, 318.0, 369.0, 420.0, &
      471.0, 522.0, 573.0, 390.0, 453.0, 516.0, 579.0, 642.0, 705.0, &
      462.0, 537.0, 612.0, 687.0, 762.0, 837.0, 534.0, 621.0, 708.0, &
      795.0, 882.0, 969.0, &
  !test 289-336
      30.0, 33.0, 36.0, 0.0, 0.0, 0.0, 102.0, 117.0, 132.0, &
      0.0, 0.0, 0.0, 174.0, 201.0, 228.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, &
  ! test 337-384
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 246.0, 264.0, &
      289.0, 0.0, 0.0, 0.0, 402.0, 432.0, 473.0, 0.0, 0.0, &
      0.0, 558.0, 600.0, 657.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, &
  ! test 385-432
      30.0, 33.0, 36.0, 39.0, 42.0, 45.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, 174.0, 201.0, 228.0, 255.0, 282.0, 309.0, &
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 318.0, 369.0, 420.0, &
      471.0, 522.0, 573.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
      462.0, 537.0, 612.0, 687.0, 762.0, 837.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, &
  ! test 433-480
      48.0, 0.0, 54.0, 0.0, 60.0, 0.0, 192.0, 0.0, 222.0, &
      0.0, 252.0, 0.0, 336.0, 0.0, 390.0, 0.0, 444.0, 0.0, &
      480.0, 0.0, 558.0, 0.0, 636.0, 0.0, 624.0, 0.0, 726.0, &
      0.0, 828.0, 0.0, 768.0, 0.0, 894.0, 0.0, 1020.0, 0.0, &
      912.0, 0.0, 1062.0, 0.0, 1212.0, 0.0, 1056.0, 0.0, 1230.0, &
      0.0, 1404.0, 0.0, &
  ! test 481-528
      30.0, 0.0, 36.0, 0.0, 42.0, 0.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, 174.0, 0.0, 228.0, 0.0, 282.0, 0.0, &
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 318.0, 0.0, 420.0, &
      0.0, 522.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
      462.0, 0.0, 612.0, 0.0, 762.0, 0.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, &
  ! test 529-576
      48.0, 0.0, 54.0, 0.0, 60.0, 0.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, 336.0, 0.0, 390.0, 0.0, 444.0, 0.0, &
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 624.0, 0.0, 726.0, &
      0.0, 828.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
      912.0, 0.0, 1062.0, 0.0, 1212.0, 0.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, &
  ! test 577-624
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 138.0, 0.0, &
      174.0, 0.0, 210.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
      0.0, 306.0, 0.0, 390.0, 0.0, 474.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, 0.0, 474.0, 0.0, 606.0, 0.0, 738.0, &
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 642.0, 0.0, &
      822.0, 0.0, 1002.0, &
  !test 625-672
      0.0, 621.0, 0.0, 795.0, 0.0, 969.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, 0.0, 453.0, 0.0, 579.0, 0.0, 705.0, &
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 285.0, 0.0, &
      363.0, 0.0, 441.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
      0.0, 117.0, 0.0, 147.0, 0.0, 177.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, &
  ! test 673-720
      912.0, 0.0, 1062.0, 0.0, 1212.0, 0.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, 624.0, 0.0, 726.0, 0.0, 828.0, 0.0, &
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 336.0, 0.0, 390.0, &
      0.0, 444.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
      48.0, 0.0, 54.0, 0.0, 60.0, 0.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, &
  ! test 721-768
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 138.0, 0.0, &
      174.0, 0.0, 210.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
      0.0, 306.0, 0.0, 390.0, 0.0, 474.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, 0.0, 474.0, 0.0, 606.0, 0.0, 738.0, &
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 642.0, 0.0, &
      822.0, 0.0, 1002.0, &
  ! test 769-816
      0.0, 90.0, 96.0, 105.0, 108.0, 114.0, 0.0, 250.0, 272.0, &
      301.0, 316.0, 338.0, 0.0, 410.0, 448.0, 497.0, 524.0, 562.0, &
      0.0, 570.0, 624.0, 693.0, 732.0, 786.0, 0.0, 730.0, 800.0, &
      889.0, 940.0, 1010.0, 0.0, 890.0, 976.0, 1085.0, 1148.0, 1234.0, &
      0.0, 1050.0, 1152.0, 1281.0, 1356.0, 1458.0, 0.0, 1210.0, 1328.0, &
      1477.0, 1564.0, 1682.0, &
  ! test 817-864
      84.0, 90.0, 96.0, 105.0, 108.0, 0.0, 228.0, 250.0, 272.0, &
      301.0, 316.0, 0.0, 372.0, 410.0, 448.0, 497.0, 524.0, 0.0, &
      516.0, 570.0, 624.0, 693.0, 732.0, 0.0, 660.0, 730.0, 800.0, &
      889.0, 940.0, 0.0, 804.0, 890.0, 976.0, 1085.0, 1148.0, 0.0, &
      948.0, 1050.0, 1152.0, 1281.0, 1356.0, 0.0, 1092.0, 1210.0, 1328.0, &
      1477.0, 1564.0, 0.0, &
  ! test 865-912
      84.0, 90.0, 96.0, 105.0, 108.0, 114.0, 228.0, 246.0, 264.0, &
      289.0, 300.0, 318.0, 372.0, 402.0, 432.0, 473.0, 492.0, 522.0, &
      516.0, 558.0, 600.0, 657.0, 684.0, 726.0, 660.0, 714.0, 768.0, &
      841.0, 876.0, 930.0, 804.0, 870.0, 936.0, 1025.0, 1068.0, 1134.0, &
      948.0, 1026.0, 1104.0, 1209.0, 1260.0, 1338.0, 1092.0, 1182.0, 1272.0, &
      1393.0, 1452.0, 1542.0, &
  ! test 913-960
      84.0, 90.0, 96.0, 105.0, 108.0, 114.0, 228.0, 246.0, 264.0, &
      289.0, 300.0, 318.0, 372.0, 402.0, 432.0, 473.0, 492.0, 522.0, &
      516.0, 558.0, 600.0, 657.0, 684.0, 726.0, 660.0, 714.0, 768.0, &
      841.0, 876.0, 930.0, 804.0, 870.0, 936.0, 1025.0, 1068.0, 1134.0, &
      948.0, 1026.0, 1104.0, 1209.0, 1260.0, 1338.0, 1092.0, 1182.0, 1272.0, &
      1393.0, 1452.0, 1542.0, &
  ! test 961-1008
      30.0, 33.0, 36.0, 39.0, 42.0, 45.0, 102.0, 117.0, 132.0, &
      147.0, 162.0, 177.0, 174.0, 201.0, 228.0, 255.0, 282.0, 309.0, &
      246.0, 285.0, 324.0, 363.0, 402.0, 441.0, 318.0, 369.0, 420.0, &
      471.0, 522.0, 573.0, 390.0, 453.0, 516.0, 579.0, 642.0, 705.0, &
      462.0, 537.0, 612.0, 687.0, 762.0, 837.0, 534.0, 621.0, 708.0, &
      795.0, 882.0, 969.0, &
  !test 1009-1056
      30.0, 33.0, 36.0, 0.0, 0.0, 0.0, 102.0, 117.0, 132.0, &
      0.0, 0.0, 0.0, 174.0, 201.0, 228.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
      0.0, 0.0, 0.0 /
  
  one = 1
  two = 2
  four = 4
  mminus1 = m_extent - 1
  nminus1 = n_extent - 1

  arr1_ptr => arr1
  arr2_ptr => arr2
  arr3_ptr => arr3

  d1_inst%arr1_ptr => arr1
  d2_inst%arr2 = arr2
  d3a_inst%arr3 = arr3
  d3b_inst%arr3_ptr => arr3

  do i = 1, m_extent
    do j = 1, k_extent
        d5_inst(i,j)%r = arr2(i,j)
    end do
  end do

  !test 1-48
  arr3_ptr=0
  arr3_ptr = matmul(arr1_ptr,arr2_ptr)
  call assign_result(1,48,arr3_ptr,results)
  ! print *,arr3_ptr
  
  ! test 49-96
  arr3_ptr=0
  arr3_ptr(2:n_extent,:) = matmul(arr1_ptr(two:n_extent,:),arr2_ptr)
  call assign_result(49,96,arr3_ptr,results)
  ! print *,arr3_ptr
  
  ! test 97-144
  arr3_ptr=0
  arr3_ptr(1:nminus1,:) = matmul(arr1_ptr(1:n_extent-1,:),arr2_ptr)
  call assign_result(97,144,arr3_ptr,results)
  ! print *,arr3_ptr
  
  ! test 145-192
  arr3_ptr=0
  arr3_ptr = matmul(arr1_ptr(:,2:m_extent),arr2_ptr(2:m_extent,:))
  call assign_result(145,192,arr3_ptr,results)
  ! print *,arr3_ptr
  
  ! test 193-240
  arr3_ptr=0
  arr3_ptr = matmul(arr1_ptr(:,2:m_extent),arr2_ptr(2:m_extent,:))
  call assign_result(193,240,arr3_ptr,results)
  ! print *,arr3_ptr
  
  ! test 241-288
  arr3_ptr=0
  arr3_ptr = matmul(arr1_ptr(:,one:mminus1),arr2_ptr(1:mminus1,:))
  call assign_result(241,288,arr3_ptr,results)
  ! print *,arr3_ptr
  
  !test 289-336
  arr3_ptr=0
  arr3_ptr(1:3,1:3) = matmul(arr1_ptr(one:3,1:3),arr2_ptr(1:3,one:3))
  call assign_result(289,336,arr3_ptr,results)
  ! print *,arr3_ptr
  
  ! test 337-384
  arr3_ptr=0
  arr3_ptr(two:4,2:four) = matmul(arr1(2:4,2:four),arr2(2:4,two:four))
  call assign_result(337,384,arr3_ptr,results)
  ! print *,arr3_ptr
  
  ! test 385-432
  arr3_ptr=0
  arr3_ptr(:,1:k_extent:2) = matmul(arr1_ptr(:,1:m_extent-1),arr2_ptr(1:m_extent-1,1:k_extent:2))
  call assign_result(385,432,arr3_ptr,results)
  ! print *,arr3_ptr
  
  ! test 433-480
  arr3_ptr=0
  arr3_ptr(1:n_extent:2,:) = matmul(arr1_ptr(1:n_extent:2,2:m_extent),arr2_ptr(1:m_extent-1,:))
  call assign_result(433,480,arr3_ptr,results)
  ! print *,arr3_ptr
  
  ! test 481-528
  arr3_ptr=0
  arr3_ptr(1:n_extent:2,1:k_extent:2) = matmul(arr1(1:n_extent:2,1:m_extent-1),      &
                                           arr2_ptr(1:m_extent-1,1:k_extent:2))
  call assign_result(481,528,arr3_ptr,results)
  ! print *,arr3_ptr
  
  ! test 529-576
  arr3_ptr=0
  arr3_ptr(1:n_extent-1:2,1:k_extent-1:2) = matmul(arr1_ptr(1:n_extent-1:2,2:m_extent),	&
                                               arr2_ptr(1:m_extent-1,1:k_extent:2))
  call assign_result(529,576,arr3_ptr,results)
  ! print *,arr3_ptr
  
  ! test 577-624
  arr3_ptr=0
  arr3_ptr(2:n_extent:2,2:k_extent:2) = matmul(arr1_ptr(2:n_extent:2,1:m_extent-1),	&
                                               arr2(2:m_extent,2:k_extent:2))
  call assign_result(577,624,arr3_ptr,results)
  ! print *,arr3_ptr
  
  !test 625-672
  arr3_ptr=0
  arr3_ptr(n_extent:1:-2,1:k_extent:2) = matmul(arr1_ptr(n_extent:1:-2,1:m_extent-1),      &
                                           arr2_ptr(1:m_extent-1,k_extent:1:-2))
  call assign_result(625,672,arr3_ptr,results)
  ! print *,arr3_ptr
  
  ! test 673-720
  arr3_ptr=0
  arr3_ptr(1:n_extent-1:2,k_extent-1:1:-2) = matmul(arr1_ptr(1:n_extent-1:2,m_extent:2:-1),	&
                                               arr2_ptr(m_extent-1:1:-1,1:k_extent:2))
  call assign_result(673,720,arr3_ptr,results)
  ! print *,arr3_ptr
  
  ! test 721-768
  arr3_ptr=0
  arr3(n_extent:2:-2,k_extent:2:-2) = matmul(arr1_ptr(n_extent:2:-2,m_extent-1:1:-1),	&
                                               arr2_ptr(m_extent:2:-1,k_extent:2:-2))
  call assign_result(721,768,arr3,results)
  ! print *,arr3_ptr
  
  ! test 769-816
  d3b_inst%arr3_ptr=0
  d3b_inst%arr3_ptr(2:n_extent,:) = matmul(arr1_ptr(two:n_extent,:),d2_inst%arr2)
  call assign_result(769,816,d3b_inst%arr3_ptr,results)
  !print *,d3b_inst%arr3_ptr
  
  ! test 817-864
  d3a_inst%arr3=0
  d3a_inst%arr3(1:nminus1,:) = matmul(d1_inst%arr1_ptr(1:n_extent-1,:),arr2_ptr)
  call assign_result(817,864,d3a_inst%arr3,results)
  ! print *,arr3_ptr
  
  ! test 865-912
  arr3_ptr=0
  arr3_ptr = matmul(d1_inst%arr1_ptr(:,2:m_extent),d2_inst%arr2(2:m_extent,:))
  call assign_result(865,912,arr3_ptr,results)
  ! print *,arr3_ptr
  
  ! test 913-960
  d3a_inst%arr3=0
  d3a_inst%arr3 = matmul(d1_inst%arr1_ptr(:,2:m_extent),arr2_ptr(2:m_extent,:))
  call assign_result(913,960,d3a_inst%arr3,results)
  ! print *,arr3_ptr
   
  ! test 961-1008
  d4_inst(:,:)%r=0
  d4_inst(:,:)%r = matmul(arr1_ptr(:,one:mminus1),arr2_ptr(1:mminus1,:))
  call assign_result(961,1008,d4_inst(:,:)%r,results)
  ! print *,d4_inst(:,:)%r
  
  !test 1009-1056
  d4_inst%r = 0
  d4_inst(1:3,1:3)%r = matmul(arr1_ptr(one:3,1:3),d5_inst(1:3,one:3)%r)
  call assign_result(1009,1056,d4_inst(:,:)%r,results)
  ! print *,d4_inst(:,:)%r
  
  call checkd(results, expect, NbrTests)
end program

subroutine assign_result(s_idx, e_idx , arr, rslt)
  REAL*8, dimension(1:e_idx-s_idx+1) :: arr
  REAL*8, dimension(e_idx) :: rslt
  integer:: s_idx, e_idx

  rslt(s_idx:e_idx) = arr

end subroutine
