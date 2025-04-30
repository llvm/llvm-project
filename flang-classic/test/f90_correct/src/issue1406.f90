! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

program issue1406
  integer, parameter :: n = 4 * 8
  integer, parameter :: expect(n) = (/ &
     5,  5,  5,  5,  5,  5,  5,  5, &
    10, 10, 10, 10, 10, 10, 10, 10, &
    15, 15, 15, 15, 15, 15, 15, 15, &
    20, 20, 20, 20, 20, 20, 20, 20  &
  /)
  type t
    integer :: a(8)
  end type
  type(t) :: stdm1(4, 5)
  integer, dimension (1:2) :: rowmajor = (/ 2, 1 /)
  type(t) :: array1(20)
  integer :: i
  do i = 1, 20
    array1(i)%a = i
  end do
  stdm1 = reshape(array1, (/ 4, 5 /), order = rowmajor)
  call check(stdm1(:, 5), expect, n)
end
