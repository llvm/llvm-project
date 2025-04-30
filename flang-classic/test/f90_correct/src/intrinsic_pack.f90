!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This test case is test for PACK intrinsic function with a scalar mask whose value is .false..

program intrinsic_pack
  integer, dimension(3, 3) :: arr1
  integer, dimension(9) :: arr2
  logical :: l1

  l1 = .false.
  arr2 = (/1, 2, 3, 4, 5, 6, 7, 8, 9/)
  arr1 = reshape(arr2, (/3, 3/))

  if (size(pack(arr1, .false.)) .ne. 0) STOP 1
  if (size(pack(arr1, l1)) .ne. 0) STOP 2
  if (size(pack(arr1(1:2, 2:3), .false.)) .ne. 0) STOP 3
  if (size(pack(arr1(1:2, 2:3), l1)) .ne. 0) STOP 4

  if (size(pack(arr1, .false., (/10, 11, 12/))) .ne. 3) STOP 5
  if (any(pack(arr1, .false., (/10, 11, 12/)) .ne. (/10, 11, 12/))) STOP 6

  if (size(pack(arr1, l1, (/10, 11, 12/))) .ne. 3) STOP 7
  if (any(pack(arr1, l1, (/10, 11, 12/)) .ne. (/10, 11, 12/))) STOP 8

  if (size(pack(arr1(1:2, 2:3), .false., (/10, 11, 12/))) .ne. 3) STOP 9
  if (any(pack(arr1(1:2, 2:3), .false., (/10, 11, 12/)) .ne. (/10, 11, 12/))) STOP 10

  if (size(pack(arr1(1:2, 2:3), l1, (/10, 11, 12/))) .ne. 3) STOP 11
  if (any(pack(arr1(1:2, 2:3), l1, (/10, 11, 12/)) .ne. (/10, 11, 12/))) STOP 12

  write(*,*) "PASS"
end program
