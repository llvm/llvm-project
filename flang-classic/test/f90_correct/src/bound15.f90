!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test the fix for the LBOUND/UBOUND regression, where the subscript expression
! as the ARRAY argument has a scalar index in some dimension.

program test
  implicit none
  integer :: arr1(2:4, 3:8)
  integer, pointer :: arr2(:, :, :)
  integer, allocatable :: res(:)

  res = ubound(arr1(3, 4:7))
  if (size(res) /= 1 .or. res(1) /= 4) STOP 1
  res = lbound(arr1(3, 4:7))
  if (size(res) /= 1 .or. res(1) /= 1) STOP 2
  allocate(arr2(2:4, 3:8, 4:10))
  res = ubound(arr2(3, 4:7, 5))
  if (size(res) /= 1 .or. res(1) /= 4) STOP 3
  res = lbound(arr2(3, 4:7, 5))
  if (size(res) /= 1 .or. res(1) /= 1) STOP 4
  print *, "PASS"
end program
