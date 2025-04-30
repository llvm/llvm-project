! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! check real conversion take KIND argument is 16

program test
  use check_mod
  real(16) :: result, expect 
  real(4) :: r4

  r4 = 1.23456_4
  result = real(r4, kind = 16)
  expect = 1.23456_4
  call checkr16(result, expect, 1)

end program
