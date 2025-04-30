!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!* Tests for runtime library MATMUL routines

program p

  parameter(NbrTests=21)

  real*4, dimension(4,3) :: arr1
  real*4, dimension(4) :: arr2
  real*4, dimension(3) :: arr3

  REAL*4 :: expect(NbrTests)
  REAL*4 :: results(NbrTests)

  data arr1 /0,1,2,3,4,5,6,7,8,9,10,11/
  data arr2 /0,1,2,3/
  data arr3 /0,1,2/

  data expect / &
 ! tests 1-3
    14.0, 38.0, 62.0,     &
 ! tests 4-6
    14.0, 38.0, 62.0,     &
 ! tests 7-9
    14.0, 38.0, 62.0,     &
 ! tests 10-12
    5.0, 17.0, 29.0,     &
 ! tests 13-15
    14.0, 38.0,  0.0,    &
 ! tests 16-18
    0.0, 38.0, 62.0,     &
 ! tests 17-21
    14.0, 38.0,  0.0/

 !print *,"tests 1-3"
  arr3=0
  arr3 = matmul(transpose(arr1),arr2)
  call assign_result(1,3,arr3,results)
 !print *,arr3

 !print *,"tests 4-6"
  arr3=0
  arr3 = matmul(transpose(arr1(2:4,:)),arr2(2:4))
  call assign_result(4,6,arr3,results)
 !print *,arr3

 !print *,"tests 7-9"
  arr3=0
  arr3 = matmul(transpose(arr1(2:4,:)),arr2(2:4))
  call assign_result(7,9,arr3,results)
 !print *,arr3

 !print *,"tests 10-12"
  arr3=0
  arr3 = matmul(transpose(arr1(1:3,:)),arr2(1:3))
  call assign_result(10,12,arr3,results)
 !print *,arr3

 !print *,"tests 13-15"
  arr3=0
  arr3(1:2) = matmul(transpose(arr1(2:4,1:2)),arr2(2:4))
  call assign_result(13,15,arr3,results)
 !print *,arr3

 !print *,"tests 16-18"
  arr3=0
  arr3(2:3) = matmul(transpose(arr1(:,2:3)),arr2)
  call assign_result(16,18,arr3,results)
 !print *,arr3

 !print *,"tests 19-21"
  arr3=0
  arr3(1:2) = matmul(transpose(arr1(:,1:2)),arr2)
  call assign_result(19,21,arr3,results)
 !print *,arr3

  call check(results, expect, NbrTests)

  
end program

subroutine assign_result(s_idx, e_idx , arr, rslt)
  REAL*4, dimension(1:e_idx-s_idx+1) :: arr
  REAL*4, dimension(e_idx) :: rslt
  integer:: s_idx, e_idx

  rslt(s_idx:e_idx) = arr

end subroutine

