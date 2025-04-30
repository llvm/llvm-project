!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!* Tests for runtime library MATMUL routines

program p
  use check_mod
  integer, parameter :: NbrTests = 210

  real*16, dimension(11, 10) :: arr1
  real*16, dimension(11) :: arr2
  real*16, dimension(10) :: arr3
  real*16, dimension(2, 10) :: arr4

  real*16 :: expect(NbrTests)
  real*16 :: results(NbrTests)

  integer :: i, j

  do i = 1, 11
    arr2(i) = i * 1.0_16
    do j = 1, 10
      arr1(i, j) = (11 * i + j - 11) * 1.0_16
    end do
  end do

  data expect / &
 4906.0_16, 4972.0_16, 5038.0_16, 5104.0_16, 5170.0_16, 5236.0_16, 5302.0_16, 5368.0_16, 5434.0_16, 5500.0_16, &
 3684.0_16, 3738.0_16, 3792.0_16, 3846.0_16, 3900.0_16, 3954.0_16, 4008.0_16, 4062.0_16, 4116.0_16, 4170.0_16, &
  229.0_16,  238.0_16,  247.0_16,  256.0_16,  265.0_16,  274.0_16,  283.0_16,  292.0_16,  301.0_16,  310.0_16, &
 2685.0_16, 2730.0_16, 2775.0_16, 2820.0_16, 2865.0_16, 2910.0_16, 2955.0_16, 3000.0_16, 3045.0_16, 3090.0_16, &
 2120.0_16, 2150.0_16, 2180.0_16, 2210.0_16, 2240.0_16, 2270.0_16, 2300.0_16, 2330.0_16,    0.0_16,    0.0_16, &
    0.0_16, 4972.0_16, 5038.0_16, 5104.0_16, 5170.0_16, 5236.0_16, 5302.0_16, 5368.0_16, 5434.0_16, 5500.0_16, &
 4906.0_16, 4972.0_16, 5038.0_16, 5104.0_16, 5170.0_16, 5236.0_16, 5302.0_16, 5368.0_16,    0.0_16,    0.0_16, &
    0.0_16, 4906.0_16,    0.0_16, 4972.0_16,    0.0_16, 5038.0_16,    0.0_16, 5104.0_16,    0.0_16, 5170.0_16, &
    0.0_16, 5236.0_16,    0.0_16, 5302.0_16,    0.0_16, 5368.0_16,    0.0_16, 5434.0_16,    0.0_16, 5500.0_16, &
    0.0_16,  229.0_16,    0.0_16,  238.0_16,    0.0_16,  247.0_16,    0.0_16,  256.0_16,    0.0_16,  265.0_16, &
    0.0_16,  274.0_16,    0.0_16,  283.0_16,    0.0_16,  292.0_16,    0.0_16,  301.0_16,    0.0_16,  310.0_16, &
    0.0_16,  229.0_16,    0.0_16,  238.0_16,    0.0_16,  247.0_16,    0.0_16,  256.0_16,    0.0_16,  265.0_16, &
    0.0_16,  274.0_16,    0.0_16,  283.0_16,    0.0_16,  292.0_16,    0.0_16,  301.0_16,    0.0_16,  310.0_16, &
    0.0_16,   94.0_16,    0.0_16,  100.0_16,    0.0_16,  106.0_16,    0.0_16,  112.0_16,    0.0_16,  118.0_16, &
    0.0_16,  124.0_16,    0.0_16,  130.0_16,    0.0_16,  136.0_16,    0.0_16,  142.0_16,    0.0_16,  148.0_16, &
    0.0_16, 2120.0_16,    0.0_16, 2150.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16, &
    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16, &
    0.0_16,    0.0_16,    0.0_16, 4972.0_16,    0.0_16, 5038.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16, &
    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16, &
    0.0_16, 4906.0_16,    0.0_16, 4972.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16, &
    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16,    0.0_16/

  arr3 = 0.0_16
  arr3 = matmul(transpose(arr1), arr2)
  call assign_result(1, 10, arr3, results)

  arr3 = 0.0_16
  arr3 = matmul(transpose(arr1(2:10, :)), arr2(2:10))
  call assign_result(11, 20, arr3, results)

  arr3 = 0.0_16
  arr3 = matmul(transpose(arr1(2:4, :)), arr2(2:4))
  call assign_result(21, 30, arr3, results)

  arr3 = 0.0_16
  arr3 = matmul(transpose(arr1(1:9, :)), arr2(1:9))
  call assign_result(31, 40, arr3, results)

  arr3 = 0.0_16
  arr3(1:8) = matmul(transpose(arr1(2:10:2, 1:8)), arr2(2:10:2))
  call assign_result(41, 50, arr3, results)

  arr3 = 0.0_16
  arr3(2:10) = matmul(transpose(arr1(:, 2:10)), arr2)
  call assign_result(51, 60, arr3, results)

  arr3 = 0.0_16
  arr3(1:8) = matmul(transpose(arr1(:, 1:8)), arr2)
  call assign_result(61, 70, arr3, results)

  arr4 = 0.0_16
  arr4(2, :) = matmul(transpose(arr1), arr2)
  call assign_result(71, 90, arr4, results)

  arr4 = 0.0_16
  arr4(2, :) = matmul(transpose(arr1(2:4, :)), arr2(2:4))
  call assign_result(91, 110, arr4, results)

  arr4 = 0.0_16
  arr4(2, :) = matmul(transpose(arr1(2:4, :)), arr2(2:4))
  call assign_result(111, 130, arr4, results)

  arr4 = 0.0_16
  arr4(2, :) = matmul(transpose(arr1(1:3, :)), arr2(1:3))
  call assign_result(131, 150, arr4, results)

  arr4 = 0.0_16
  arr4(2, 1:2) = matmul(transpose(arr1(2:10:2, 1:2)), arr2(2:10:2))
  call assign_result(151, 170, arr4, results)

  arr4 = 0.0_16
  arr4(2, 2:3) = matmul(transpose(arr1(:, 2:3)), arr2)
  call assign_result(171, 190, arr4, results)

  arr4 = 0.0_16
  arr4(2, 1:2) = matmul(transpose(arr1(:, 1:2)), arr2)
  call assign_result(191, 210, arr4, results)

  call checkr16(results, expect, NbrTests)
end program

subroutine assign_result(s_idx, e_idx, arr, rslt)
  integer:: s_idx, e_idx
  real*16, dimension(1:e_idx - s_idx + 1) :: arr
  real*16, dimension(e_idx) :: rslt

  rslt(s_idx:e_idx) = arr
end subroutine

