!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!* Tests for runtime library MATMUL routines

program p

  parameter(NbrTests=172)

  integer*8, dimension(4,3) :: arr1
  integer*8, dimension(3) :: arr2
  integer*8, dimension(4) :: arr3
  integer*8, dimension(4,4) :: arr4
  integer*8, dimension(0:3,-1:1) :: arr5
  integer*8, dimension(-3:-1) :: arr6
  integer*8, dimension(-1:2,0:3) :: arr7

  data arr1 /0,1,2,3,4,5,6,7,8,9,10,11/
  data arr5 /0,1,2,3,4,5,6,7,8,9,10,11/
  data arr2 /0,1,2/
  data arr6 /0,1,2/
  data arr3 /0,1,2,3/
  data arr4 /0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15/
  data arr7 /0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15/

  integer*8 :: expect(NbrTests)
  integer*8 :: results(NbrTests)

  data expect /  &
  ! tests 1-4
      20, 23, 26, 29,  &
  ! tests 5-8
      0, 23, 26, 29,  &
  ! tests 9-12
      23, 26, 29, 0,  &
  ! tests 13-16
      0, 20, 23, 26,  &
  ! tests 17-20
      0, 5, 6, 7,  &
  ! tests 21-24
      8, 9, 10, 11,  &
  ! tests 25-28
      8, 11, 14, 17,  &
  ! tests 29-32
      0, 8, 11, 14,  &
  ! tests 33-36
      9, 10, 11, 0,  &
  ! tests 37-40
      18, 20, 22, 0,  &
  ! tests 41-44
      0, 8, 0, 14,  &
  ! tests 45-48
      9, 0, 11,  &
      0,  &
  ! tests 49-64
      0, 9, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0,  &
      0, 0, 0, 0,  &
  ! tests 65-80
      0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 11, 0,  &
      0, 0, 0, 0,  &
  ! tests 81-96
      0, 9, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0,  &
      0, 0, 0, 0,  &
  ! tests 97-112
      0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 29, 0,  &
      0, 0, 0, 0,  &
  ! tests 113-128
      22, 20, 18, 0,  &
  ! tests 129-145
      9, 0, 11, 0,  &
  ! tests 145-160
      0, 11, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0,  &
      0, 0, 0, 0,  &
  ! tests 161-176
      0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 11, 0,  &
      0, 0, 0, 0,  &
  ! tests 177-192
      0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 29, 0,  &
      0, 0, 0, 0,  &
  ! tests 193-196
      18, 20, 22, 0 /

  results = -1

  ! tests 1-4
  arr3=0
  arr3 = matmul(arr1,arr2)
  call assign_result(1,4,arr3,results)
  !print *,arr3
  
  ! tests 5-8
  arr3=0
  arr3(2:4) = matmul(arr1(2:4,:),arr2)
  call assign_result(5,8,arr3,results)
  !print *,arr3
  
  ! tests 9-12
  arr3=0
  arr3(1:3) = matmul(arr1(2:4,:),arr2)
  call assign_result(9,12,arr3,results)
  !print *,arr3
  
  !tests 13-16
  arr3=0
  arr3(2:4) = matmul(arr1(1:3,:),arr2)
  call assign_result(13,16,arr3,results)
  !print *,arr3
  
  !tests 17-20
  arr3=0
  arr3(2:4) = matmul(arr1(2:4,1:2),arr2(1:2))
  call assign_result(17,20,arr3,results)
  !print *,arr3
  
  !tests 21-24
  arr3=0
  arr3 = matmul(arr1(:,2:3),arr2(1:2))
  call assign_result(21,24,arr3,results)
  !print *,arr3
  
  !tests 25-28
  arr3=0
  arr3 = matmul(arr1(:,1:2),arr2(2:3))
  call assign_result(25,28,arr3,results)
  !print *,arr3
  
  !tests 29-32
  arr3=0
  arr3(2:4)  = matmul(arr1(1:3,1:2),arr2(2:3))
  call assign_result(29,32,arr3,results)
  !print *,arr3
  
  !tests 33-36
  arr3=0
  arr3(1:3)  = matmul(arr1(2:4,2:3),arr2(1:2))
  call assign_result(33,36,arr3,results)
  !print *,arr3
  
  !tests 37-40
  arr3=0
  arr3(1:3) = matmul(arr1(2:4,1:3:2),arr2(1:3:2))
  call assign_result(37,40,arr3,results)
  !print *,arr3
  
  !tests 41-44
  arr3=0
  arr3(2:4:2)  = matmul(arr1(1:3:2,1:2),arr2(2:3))
  call assign_result(41,44,arr3,results)
  !print *,arr3
  
  !tests 45-48
  arr3=0
  arr3(1:3:2)  = matmul(arr1(2:4:2,2:3),arr2(1:2))
  call assign_result(45,48,arr3,results)
  !print *,arr3
  
  !tests 49-64
  arr4=0
  arr4(2,1:3:2)  = matmul(arr1(2:4:2,2:3),arr2(1:2))
  call assign_result(49,64,arr4,results)
  !print *,arr4
  
  !tests 65-80
  arr4=0
  arr4(1:3:2,3)  = matmul(arr1(2:4:2,2:3),arr2(1:2))
  call assign_result(65,80,arr4,results)
  !print *,arr4
  
  !tests 81-96
  arr7=0
  arr7(0,0:2:2)  = matmul(arr5(1:3:2,0:1),arr6(-3:-2))
  call assign_result(81,96,arr7,results)
  !print *,arr7
  
  !tests 97-112
  arr7=0
  arr7(-1:1:2,2)  = matmul(arr5(1:3:2,0:1),arr6(-2:-1))
  call assign_result(97,112,arr7,results)
  !print *,arr7
  
  !tests 113-116
  arr3=0
  arr3(3:1:-1) = matmul(arr1(2:4,3:1:-2),arr2(3:1:-2))
  call assign_result(113,116,arr3,results)
  !print *,arr3
  
  !tests 117-120
  arr3=0
  arr3(3:1:-2)  = matmul(arr1(4:2:-2,2:3),arr2(1:2))
  call assign_result(117,120,arr3,results)
  !print *,arr3
  
  !tests 121,136
  arr4=0
  arr4(2,3:1:-2)  = matmul(arr1(2:4:2,2:3),arr2(1:2))
  call assign_result(121,136,arr4,results)
  !print *,arr4
  
  !tests 137-152
  arr4=0
  arr4(3:1:-2,3)  = matmul(arr1(4:2:-2,2:3),arr2(1:2))
  call assign_result(137,152,arr4,results)
  !print *,arr4
  
  !tests 153-168
  arr7=0
  arr7(1:-1:-2,2)  = matmul(arr5(3:1:-2,0:1),arr6(-2:-1))
  call assign_result(153,168,arr7,results)
  !print *,arr7
  
  !tests 169-172
  arr3=0
  arr3(1:3) = matmul(arr1(2:4,3:1:-2),arr2(3:1:-2))
  call assign_result(169,172,arr3,results)
  !print *,arr3

  call checkll(results, expect, NbrTests)

end program

subroutine assign_result(s_idx, e_idx , arr, rslt)
  integer*8, dimension(1:e_idx-s_idx+1) :: arr
  integer*8, dimension(e_idx) :: rslt
  integer:: s_idx, e_idx

  rslt(s_idx:e_idx) = arr

end subroutine

