! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! this test case is test for assigning values for quadruple precission array

program test
use check_mod
  real(16), dimension(3,3) :: a, b, c
  a = 999999999.8888_16
  b = a
  c = 0.0_16

  call checkr16(a-b, c, 9)
end program test
