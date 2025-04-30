! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for quad precision

program test
  use check_mod

  real(16) :: result1(10), result2(10), expect1(10), expect2(10) ,r(20), er(20)
  data result1 / 10*0.0_16 /
  data result2 / 10*1.0_16 /

  expect1 = 0.0_16
  expect2 = 1.0_16

  r(1:10) = result1
  r(11:20) = result2
  er(1:10) = expect1
  er(11:20) = expect2

  call checkr16(r, er, 20)
end
