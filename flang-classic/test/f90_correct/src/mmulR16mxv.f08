!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!* Tests for runtime library MATMUL routines

program p
  use check_mod
  integer, parameter :: NbrTests = 300

  real*16, dimension(10, 9) :: arr1
  real*16, dimension(9) :: arr2
  real*16, dimension(10) :: arr3
  real*16, dimension(4, 4) :: arr4
  real*16, dimension(0:3, -1:1) :: arr5
  real*16, dimension(-3:-1) :: arr6
  real*16, dimension(-1:2, 0:3) :: arr7
  real*16, dimension(2:5, 3) :: arr8
  real*16, dimension(2:4) :: arr9
  real*16, dimension(2:5) :: arr10
  real*16, dimension(4, 2:4) :: arr11
  real*16, dimension(1:4) :: arr12
  real*16 :: expect(NbrTests)
  real*16 :: results(NbrTests)

  integer :: i, j

! data arr1 /0.0_16,1.0_16,2.0_16,3.0_16,4.0_16,5.0_16, &
!            6.0_16,7.0_16,8.0_16,9.0_16,10.0_16,11.0_16/
! data arr2 /0.0_16,1.0_16,2.0_16/
  do i = 1, 9
    arr2(i) = i * 1.0_16
    do j = 1, 10
      arr1(j, i) = (10 * j + i - 10) * 1.0_16
    end do
  end do

  data arr5 /0.0_16,1.0_16,2.0_16,3.0_16,4.0_16,5.0_16,   &
             6.0_16,7.0_16,8.0_16,9.0_16,10.0_16,11.0_16/
  data arr6 /0.0_16,1.0_16,2.0_16/
  data arr4 /0.0_16,1.0_16,2.0_16,3.0_16,4.0_16,5.0_16,   &
             6.0_16,7.0_16,8.0_16,9.0_16,10.0_16,11.0_16, &
             12.0_16,13.0_16,14.0_16,15.0_16/
  data arr7 /0.0_16,1.0_16,2.0_16,3.0_16,4.0_16,5.0_16,   &
             6.0_16,7.0_16,8.0_16,9.0_16,10.0_16,11.0_16, &
             12.0_16,13.0_16,14.0_16,15.0_16/
  data arr8 /0.0_16,1.0_16,2.0_16,3.0_16,4.0_16,5.0_16,   &
             6.0_16,7.0_16,8.0_16,9.0_16,10.0_16,11.0_16/
  data arr9 /0.0_16,1.0_16,2.0_16/
  data arr10 /0.0_16,1.0_16,2.0_16,3.0_16/
  data arr11 /0.0_16,1.0_16,2.0_16,3.0_16,4.0_16,5.0_16,  &
              6.0_16,7.0_16,8.0_16,9.0_16,10.0_16,11.0_16/

  data expect/ &
  285.0_16, 735.0_16,1185.0_16,1635.0_16,2085.0_16,2535.0_16,2985.0_16,3435.0_16,3885.0_16,4335.0_16, &
    0.0_16, 735.0_16,1185.0_16,1635.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16, &
  735.0_16,1185.0_16,1635.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16, &
    0.0_16, 285.0_16, 735.0_16,1185.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16, &
    0.0_16,  35.0_16,  65.0_16,  95.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16, &
    8.0_16,  38.0_16,  68.0_16,  98.0_16, 128.0_16, 158.0_16, 188.0_16, 218.0_16, 248.0_16, 278.0_16, &
    8.0_16,  58.0_16, 108.0_16, 158.0_16, 208.0_16, 258.0_16, 308.0_16, 358.0_16, 408.0_16, 458.0_16, &
    0.0_16,   8.0_16,  58.0_16, 108.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16, &
   38.0_16,  68.0_16,  98.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16, &
   50.0_16,  90.0_16, 130.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16, &
    0.0_16,   8.0_16,   0.0_16, 108.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16, &
   38.0_16,   0.0_16,  98.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16, &
    0.0_16,  38.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,  98.0_16, &
    0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16, &
    0.0_16,   0.0_16,   0.0_16,   0.0_16,  38.0_16,   0.0_16,  98.0_16,   0.0_16,   0.0_16,   0.0_16, &
    0.0_16,   0.0_16,   0.0_16,   9.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16, &
    0.0_16,  11.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16, &
    0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,  23.0_16,   0.0_16,  29.0_16,   0.0_16, &
    0.0_16,   0.0_16,   0.0_16,   0.0_16, 130.0_16,  90.0_16,  50.0_16,   0.0_16,   0.0_16,   0.0_16, &
    0.0_16,   0.0_16,   0.0_16,   0.0_16,  38.0_16,   0.0_16,  98.0_16,   0.0_16,   0.0_16,   0.0_16, &
    0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,  98.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16, &
    0.0_16,   0.0_16,   0.0_16,  38.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16, &
    0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,  38.0_16,   0.0_16, &
   98.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16, &
    0.0_16,   0.0_16,   0.0_16,   0.0_16,  23.0_16,   0.0_16,  29.0_16,   0.0_16,   0.0_16,   0.0_16, &
    0.0_16,   0.0_16,  50.0_16,  90.0_16, 130.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16,   0.0_16, &
    0.0_16,   0.0_16,   0.0_16,  20.0_16,  23.0_16,  26.0_16,  29.0_16,  20.0_16,  23.0_16,  26.0_16, &
   29.0_16,  23.0_16,  26.0_16,  29.0_16,   0.0_16,   0.0_16,  20.0_16,  23.0_16,  26.0_16,   0.0_16, &
    5.0_16,   6.0_16,   7.0_16,   8.0_16,   9.0_16,  10.0_16,  11.0_16,   8.0_16,  11.0_16,  14.0_16, &
    0.0_16,   8.0_16,  11.0_16,  14.0_16,   9.0_16,  10.0_16,  11.0_16,   0.0_16,  -1.0_16,  -1.0_16/

  results = -1.0_16

  ! tests 1-4
  arr3 = 0.0_16
  arr3 = matmul(arr1, arr2)
  call assign_result(1, 10, arr3, results)

  ! tests 5-8
  arr3 = 0.0_16
  arr3(2:4) = matmul(arr1(2:4, :), arr2)
  call assign_result(11, 20, arr3, results)

  ! tests 9-12
  arr3 = 0.0_16
  arr3(1:3) = matmul(arr1(2:4, :), arr2)
  call assign_result(21, 30, arr3, results)

  !tests 13-16
  arr3 = 0.0_16
  arr3(2:4) = matmul(arr1(1:3, :), arr2)
  call assign_result(31, 40, arr3, results)

  !tests 17-20
  arr3 = 0.0_16
  arr3(2:4) = matmul(arr1(2:4, 1:2), arr2(1:2))
  call assign_result(41, 50, arr3, results)

  !tests 21-24
  arr3 = 0.0_16
  arr3 = matmul(arr1(:, 2:3), arr2(1:2))
  call assign_result(51, 60, arr3, results)

  !tests 25-28
  arr3 = 0.0_16
  arr3 = matmul(arr1(:, 1:2), arr2(2:3))
  call assign_result(61, 70, arr3, results)

  !tests 29-32
  arr3 = 0.0_16
  arr3(2:4) = matmul(arr1(1:3, 1:2), arr2(2:3))
  call assign_result(71, 80, arr3, results)

  !tests 33-36
  arr3 = 0.0_16
  arr3(1:3) = matmul(arr1(2:4, 2:3), arr2(1:2))
  call assign_result(81, 90, arr3, results)

  !tests 37-40
  arr3 = 0.0_16
  arr3(1:3) = matmul(arr1(2:4, 1:3:2), arr2(1:3:2))
  call assign_result(91, 100, arr3, results)

  !tests 41-44
  arr3 = 0.0_16
  arr3(2:4:2) = matmul(arr1(1:3:2, 1:2), arr2(2:3))
  call assign_result(101, 110, arr3, results)

  !tests 45-48
  arr3 = 0.0_16
  arr3(1:3:2) = matmul(arr1(2:4:2, 2:3), arr2(1:2))
  call assign_result(111, 120, arr3, results)

  !tests 49-64
  arr4 = 0.0_16
  arr4(2, 1:3:2) = matmul(arr1(2:4:2, 2:3), arr2(1:2))
  call assign_result(121, 136, arr4, results)

  !tests 65-80
  arr4 = 0.0_16
  arr4(1:3:2, 3) = matmul(arr1(2:4:2, 2:3), arr2(1:2))
  call assign_result(137, 152, arr4, results)

  !tests 81-96
  arr7 = 0.0_16
  arr7(0, 0:2:2) = matmul(arr5(1:3:2, 0:1), arr6(-3:-2))
  call assign_result(153, 168, arr7, results)

  !tests 97-112
  arr7 = 0.0_16
  arr7(-1:1:2, 2) = matmul(arr5(1:3:2, 0:1), arr6(-2:-1))
  call assign_result(169, 184, arr7, results)

  !tests 113-116
  arr3 = 0.0_16
  arr3(3:1:-1) = matmul(arr1(2:4, 3:1:-2), arr2(3:1:-2))
  call assign_result(185, 194, arr3, results)

  !tests 117-120
  arr3 = 0.0_16
  arr3(3:1:-2) = matmul(arr1(4:2:-2, 2:3), arr2(1:2))
  call assign_result(195, 204, arr3, results)

  !tests 121,136
  arr4 = 0.0_16
  arr4(2, 3:1:-2) = matmul(arr1(2:4:2, 2:3), arr2(1:2))
  call assign_result(205, 220, arr4, results)

  !tests 137-152
  arr4 = 0.0_16
  arr4(3:1:-2, 3) = matmul(arr1(4:2:-2, 2:3), arr2(1:2))
  call assign_result(221, 236, arr4, results)

  !tests 153-168
  arr7 = 0.0_16
  arr7(1:-1:-2, 2) = matmul(arr5(3:1:-2, 0:1), arr6(-2:-1))
  call assign_result(237, 252, arr7, results)

  !tests 169-172
  arr3 = 0.0_16
  arr3(1:3) = matmul(arr1(2:4, 3:1:-2), arr2(3:1:-2))
  call assign_result(253, 263, arr3, results)

  arr12 = 0.0_16

  arr10 = 0.0_16
  arr10 = arr12 + matmul(arr8, arr9)
  call assign_result(264, 267, arr10, results)

  arr10 = 0.0_16
  arr10 = arr12 + matmul(arr11, arr9)
  call assign_result(268, 271, arr10, results)

  arr10 = 0.0_16
  arr10(2:4) = arr12(2:4) + matmul(arr8(3:5, :), arr9)
  call assign_result(272, 275, arr10, results)

  arr10 = 0.0_16
  arr10(3:5) = arr12(2:4) + matmul(arr8(2:4, :), arr9)
  call assign_result(276, 279, arr10, results)

  arr10 = 0.0_16
  arr10(3:5) = arr12(2:4) + matmul(arr8(3:5, 1:2), arr9(2:3))
  call assign_result(280, 283, arr10, results)

  arr10 = 0.0_16
  arr10 = arr12 + matmul(arr8(:, 2:3), arr9(2:3))
  call assign_result(284, 287, arr10, results)

  arr10 = 0.0_16
  arr10 = arr12 + matmul(arr8(:, 1:2), arr9(3:4))
  call assign_result(288, 291, arr10, results)

  arr10 = 0.0_16
  arr10(3:5) = arr12(2:4) + matmul(arr8(2:4, 1:2), arr9(3:4))
  call assign_result(291, 294, arr10, results)

  arr10 = 0.0_16
  arr10(2:4) = arr12(2:4) + matmul(arr8(3:5, 2:3), arr9(2:3))
  call assign_result(295, 298, arr10, results)
  call checkr16(results, expect, NbrTests)

end program

subroutine assign_result(s_idx, e_idx, arr, rslt)
  integer:: s_idx, e_idx
  real*16, dimension(1:e_idx - s_idx + 1) :: arr
  real*16, dimension(e_idx) :: rslt

  rslt(s_idx:e_idx) = arr

end subroutine

