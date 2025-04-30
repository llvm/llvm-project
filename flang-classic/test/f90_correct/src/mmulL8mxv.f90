!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!* Tests for runtime library MATMUL routines

program p

  parameter(NbrTests=176)

  logical*8, dimension(4,3) :: arr1
  logical*8, dimension(3) :: arr2
  logical*8, dimension(4) :: arr3
  logical*8, dimension(4,4) :: arr4
  logical*8, dimension(0:3,-1:1) :: arr5
  logical*8, dimension(-3:-1) :: arr6
  logical*8, dimension(-1:2,0:3) :: arr7
  logical*8, dimension(2:5,2:4) :: arr8
  logical*8, dimension(2:4) :: arr9
  logical*8, dimension(2:5) :: arr10
  logical*8, dimension(4) :: arr11


  data arr1 /.true.,.false.,.true.,.false., &
             .false.,.true.,.false.,.true., &
             .false.,.false.,.true.,.true./
  data arr2 /.true.,.false.,.true./
  data arr3 /.true.,.false.,.true.,.false./
  data arr4 /.true.,.false.,.true.,.false., &
             .false.,.true.,.false.,.true., &
             .false.,.false.,.true.,.true., &
             .true.,.true.,.false.,.false./
  data arr5 /.true.,.false.,.true.,.false., &
             .false.,.false.,.true.,.true., &
             .false.,.true.,.false.,.true./
  data arr6 /.false.,.true.,.true./
  data arr7 /.false.,.true.,.false.,.true., &
             .true.,.false.,.true.,.false., &
             .false.,.true.,.false.,.true., &
             .true.,.false.,.true.,.false./
  data arr8 /.true.,.false.,.true.,.false., &
             .false.,.true.,.false.,.true., &
             .false.,.false.,.true.,.true./
  data arr9 /.true.,.false.,.true./
  data arr10 /.true.,.false.,.true.,.false./

  logical*8 :: expect(NbrTests)
  logical*8 :: results(NbrTests)

  data expect /  &
  !test 1-4
    -1, 0, -1, -1, &
  !test 5-8
     0, 0, -1, -1, &
  !test 9-12
     0, -1, -1, 0, &
  !test 13-16
     0, -1, 0, -1, &
  !test 17-20, &
     0, 0, -1, 0, &
  !test 21-24
     0, -1, 0, -1, &
  !test 25-28
     0, -1, 0, -1, &
  !test 29-32
     0, 0, -1, 0, &
  !test 33-36
    -1, 0, -1, 0, &
  !test 37-40, &
     0, -1, -1, 0, &
  !test 41-44
     0, 0, 0, 0, &
  !test 45-48
    -1, 0, -1, 0, &
  !test 49-64
     0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, &
  !test 65-80, &
     0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, &
  !test 81-96
     0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, &
  !test 97-112
     0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, &
  !test 113-116
    -1, -1, 0, 0, &
  !test 117-120, &
    -1, 0, -1, 0, &
  !test 121-136
     0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, &
  !test 137-152
     0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, &
  !test 153-168
     0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, &
  !test 169-172
     0, -1, -1, 0, &
    !test 173-176
    -1, 0, -1, -1 /

  ! tests 1-4
  arr3=0
  arr3 = matmul(arr1,arr2)
  call assign_result(1,4,arr3,results)
  !print *,"test 1-4"
  !print *,arr3
  
  ! tests 5-8
  arr3=0
  arr3(2:4) = matmul(arr1(2:4,:),arr2)
  call assign_result(5,8,arr3,results)
  !print *,"test 5-8"
  !print *,arr3
  
  ! tests 9-12
  arr3=0
  arr3(1:3) = matmul(arr1(2:4,:),arr2)
  call assign_result(9,12,arr3,results)
  !print *,"test 9-12"
  !print *,arr3
  
  !tests 13-16
  arr3=0
  arr3(2:4) = matmul(arr1(1:3,:),arr2)
  call assign_result(13,16,arr3,results)
  !print *,"test 13-16"
  !print *,arr3
  
  !tests 17-20
  arr3=0
  arr3(2:4) = matmul(arr1(2:4,1:2),arr2(1:2))
  call assign_result(17,20,arr3,results)
  !print *,"test 17-20"
  !print *,arr3
  
  !tests 21-24
  arr3=0
  arr3 = matmul(arr1(:,2:3),arr2(1:2))
  call assign_result(21,24,arr3,results)
  !print *,"test 21-24"
  !print *,arr3
  
  !tests 25-28
  arr3=0
  arr3 = matmul(arr1(:,1:2),arr2(2:3))
  call assign_result(25,28,arr3,results)
  !print *,"test 25-28"
  !print *,arr3
  
  !tests 29-32
  arr3=0
  arr3(2:4)  = matmul(arr1(1:3,1:2),arr2(2:3))
  call assign_result(29,32,arr3,results)
  !print *,"test 29-32"
  !print *,arr3
  
  !tests 33-36
  arr3=0
  arr3(1:3)  = matmul(arr1(2:4,2:3),arr2(1:2))
  call assign_result(33,36,arr3,results)
  !print *,"test 33-36"
  !print *,arr3
  
  !tests 37-40
  arr3=0
  arr3(1:3) = matmul(arr1(2:4,1:3:2),arr2(1:3:2))
  call assign_result(37,40,arr3,results)
  !print *,"test 37-40"
  !print *,arr3
  
  !tests 41-44
  arr3=0
  arr3(2:4:2)  = matmul(arr1(1:3:2,1:2),arr2(2:3))
  call assign_result(41,44,arr3,results)
  !print *,"test 41-44"
  !print *,arr3
  
  !tests 45-48
  arr3=0
  arr3(1:3:2)  = matmul(arr1(2:4:2,2:3),arr2(1:2))
  call assign_result(45,48,arr3,results)
  !print *,"test 45-48"
  !print *,arr3
  
  !tests 49-64
  arr4=0
  arr4(2,1:3:2)  = matmul(arr1(2:4:2,2:3),arr2(1:2))
  call assign_result(49,64,arr4,results)
  !print *,"test 49-64"
  !print *,arr4
  
  !tests 65-80
  arr4=0
  arr4(1:3:2,3)  = matmul(arr1(2:4:2,2:3),arr2(1:2))
  call assign_result(65,80,arr4,results)
  !print *,"test 65-80"
  !print *,arr4
  
  !tests 81-96
  arr7=0
  arr7(0,0:2:2)  = matmul(arr5(1:3:2,0:1),arr6(-3:-2))
  call assign_result(81,96,arr7,results)
  !print *,"test 81-96"
  !print *,arr7
  
  !tests 97-112
  arr7=0
  arr7(-1:1:2,2)  = matmul(arr5(1:3:2,0:1),arr6(-2:-1))
  call assign_result(97,112,arr7,results)
  !print *,"test 97-112"
  !print *,arr7
  
  !tests 113-116
  arr3=0
  arr3(3:1:-1) = matmul(arr1(2:4,3:1:-2),arr2(3:1:-2))
  call assign_result(113,116,arr3,results)
  !print *,"test 113-116"
  !print *,arr3
  
  !tests 117-120
  arr3=0
  arr3(3:1:-2)  = matmul(arr1(4:2:-2,2:3),arr2(1:2))
  call assign_result(117,120,arr3,results)
  !print *,"test 117-120"
  !print *,arr3
  
  !tests 121,136
  arr4=0
  arr4(2,3:1:-2)  = matmul(arr1(2:4:2,2:3),arr2(1:2))
  call assign_result(121,136,arr4,results)
  !print *,"test 121-136"
  !print *,arr4
  
  !tests 137-152
  arr4=0
  arr4(3:1:-2,3)  = matmul(arr1(4:2:-2,2:3),arr2(1:2))
  call assign_result(137,152,arr4,results)
  !print *,"test 137-152"
  !print *,arr4
  
  !tests 153-168
  arr7=0
  arr7(1:-1:-2,2)  = matmul(arr5(3:1:-2,0:1),arr6(-2:-1))
  call assign_result(153,168,arr7,results)
  !print *,"test 153-168"
  !print *,arr7
  
  !tests 169-172
  arr3=0
  arr3(1:3) = matmul(arr1(2:4,3:1:-2),arr2(3:1:-2))
  call assign_result(169,172,arr3,results)
  !print *,"test 169-172"
  !print *,arr3

  arr11 = .false.

  ! tests 173-176
  arr10=0
  arr10 = arr11 .or.  matmul(arr8,arr9)
  call assign_result(173,176,arr10,results)
  !print *,"test 173-176"
  !print *,arr10

  call checkll(results, expect, NbrTests)

end program

subroutine assign_result(s_idx, e_idx , arr, rslt)
  logical*8, dimension(1:e_idx-s_idx+1) :: arr
  logical*8, dimension(e_idx) :: rslt
  integer:: s_idx, e_idx

  rslt(s_idx:e_idx) = arr

end subroutine
