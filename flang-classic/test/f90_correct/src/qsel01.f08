! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test quad-precision support in SELECTED_REAL_KIND intrinsic, with both
! constant and variable inputs

program main
  integer :: a(9)
  integer result(8), expect(8)
  data expect / 16, 16, 16, 16, -1, -2,  -3, -5 /
  data a / 16, 33, 308, 4931, 34, 4932, 6, 40, 11 /

  result(1) = selected_real_kind(p = 16)
  result(2) = selected_real_kind(p = 33)
  result(3) = selected_real_kind(r = 308)
  result(4) = selected_real_kind(r = 4931)
  result(5) = selected_real_kind(p = 34)
  result(6) = selected_real_kind(r = 4932)
  result(7) = selected_real_kind(p = 34, r = 4932)
  result(8) = selected_real_kind(p = 6, r = 40, radix = 10)

  if(any(result /= expect)) STOP 1

  result(1) = selected_real_kind(p = a(1))
  result(2) = selected_real_kind(p = a(2))
  result(3) = selected_real_kind(r = a(3))
  result(4) = selected_real_kind(r = a(4))
  result(5) = selected_real_kind(p = a(5))
  result(6) = selected_real_kind(r = a(6))
  result(7) = selected_real_kind(p = a(5), r = a(6))
  result(8) = selected_real_kind(p = a(7), r = a(8), radix = a(9))
  call check(result, expect, 8)

end program main
