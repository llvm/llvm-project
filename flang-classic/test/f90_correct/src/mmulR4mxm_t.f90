!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!* Tests for runtime library MATMUL routines

program p

  parameter(NbrTests=128)
  parameter(n_extent=6)
  parameter(m_extent=4)
  parameter(k_extent=8)

  real*4, dimension(n_extent,m_extent) :: arr1
  real*4, dimension(n_extent,k_extent) :: arr2
  real*4, dimension(m_extent,k_extent) :: arr3

  REAL*4 :: expect(NbrTests)
  REAL*4 :: results(NbrTests)
  
  integer:: i,j
  
  data arr1 /0,1,2,3,4,5,6,7,8,9,10,11,                 &
             12,13,14,15,16,17,18,19,20,22,22,23/
  data arr2 /0,1,2,3,4,5,6,7,8,9,10,11,                         &
             12,13,14,15,16,17,18,19,20,21,22,23,               &
             24,25,26,27,28,29,30,31,32,33,34,35,               &
             36,37,38,39,40,41,42,43,44,45,46,47/
  data arr3 /0,1,2,3,4,5,6,7,8,9,10,11,                 &
             12,13,14,15,16,17,18,19,20,21,22,23,               &
             24,25,26,27,28,29,30,31/

  data expect / &
 ! test 1-32
    55.0, 145.0, 235.0, 328.0, 145.0, 451.0, &
    757.0, 1072.0, 235.0, 757.0, 1279.0, 1816.0, &
    325.0, 1063.0, 1801.0, 2560.0, 415.0, 1369.0, &
    2323.0, 3304.0, 505.0, 1675.0, 2845.0, 4048.0, &
    595.0, 1981.0, 3367.0, 4792.0, 685.0, 2287.0, &
    3889.0, 5536.0, &
 ! test 43-64
    55.0, 145.0, 235.0, 328.0, 145.0, 415.0, &
    685.0, 964.0, 235.0, 685.0, 1135.0, 1600.0, &
    325.0, 955.0, 1585.0, 2236.0, 415.0, 1225.0, &
    2035.0, 2872.0, 505.0, 1495.0, 2485.0, 3508.0, &
    595.0, 1765.0, 2935.0, 4144.0, 685.0, 2035.0, &
    3385.0, 4780.0, &
 ! test 65-96
    30.0, 90.0, 150.0, 213.0, 90.0, 330.0, &
    570.0, 819.0, 150.0, 570.0, 990.0, 1425.0, &
    210.0, 810.0, 1410.0, 2031.0, 270.0, 1050.0, &
    1830.0, 2637.0, 330.0, 1290.0, 2250.0, 3243.0, &
    390.0, 1530.0, 2670.0, 3849.0, 450.0, 1770.0, &
    3090.0, 4455.0, &
 ! test 97-128
    0.0, 145.0, 235.0, 328.0, 0.0, 451.0, &
    757.0, 1072.0, 0.0, 757.0, 1279.0, 1816.0, &
    0.0, 1063.0, 1801.0, 2560.0, 0.0, 1369.0, &
    2323.0, 3304.0, 0.0, 1675.0, 2845.0, 4048.0, &
    0.0, 1981.0, 3367.0, 4792.0, 0.0, 2287.0, &
    3889.0, 5536.0/


  !print *,"test 1-32"
  arr3=0
  arr3 = matmul(transpose(arr1),arr2)
  call assign_result(1,32,arr3,results)
  !print *,arr3

  !print *,"test 33-64"
  arr3=0
  arr3 = matmul(transpose(arr1(2:n_extent,:)),arr2(2:n_extent,:))
  call assign_result(33,64,arr3,results)
  !print *,arr3

  !print *,"test 65-96 "
  arr3=0 
  arr3 = matmul(transpose(arr1(1:n_extent-1,:)),arr2(1:n_extent-1,:))
  call assign_result(65,96,arr3,results) 
  !print *,arr3

  !print *,"test 97-128"
  arr3=0
  arr3(2:m_extent,:) = matmul(transpose(arr1(:,2:m_extent)),arr2)
  call assign_result(97,128,arr3,results)
  !print *,arr3
  
  call check(results, expect, NbrTests)
  
end program

subroutine assign_result(s_idx, e_idx , arr, rslt)
  REAL*4, dimension(1:e_idx-s_idx+1) :: arr
  REAL*4, dimension(e_idx) :: rslt
  integer:: s_idx, e_idx

  rslt(s_idx:e_idx) = arr

end subroutine

