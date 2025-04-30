! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test quad-precision support in the SELECTED_REAL_KIND intrinsic when
! used in variable initializations

program main
  integer result(8), expect(8)
  integer, parameter :: i1 = selected_real_kind(p = 16)
  integer, parameter :: i2 = selected_real_kind(p = 33)
  integer, parameter :: i3 = selected_real_kind(r = 308)
  integer, parameter :: i4 = selected_real_kind(r = 4931)
  integer, parameter :: i5 = selected_real_kind(p = 34)
  integer, parameter :: i6 = selected_real_kind(r = 4932)
  integer, parameter :: i7 = selected_real_kind(p = 34, r = 4932)
  integer, parameter :: i8 = selected_real_kind(p = 6, r = 40, radix = 10)
  data expect / 16, 16, 16, 16, -1, -2,  -3, -5 /

  result(1) = i1
  result(2) = i2
  result(3) = i3
  result(4) = i4
  result(5) = i5
  result(6) = i6
  result(7) = i7
  result(8) = i8

  call check(result, expect, 8)

end program main
