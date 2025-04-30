! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for cosd take quad precision argument

program test
  real(16) :: r1, r2, q1 = -1.23456789_16
  real(16) :: eps_q = 1.e-33_16
  integer :: result(2), expect(2)
  expect = 1
  result = 0

  r1 = cosd(9.87654321_16)
  r2 = cosd(q1)
  
  if (abs((r1 - 0.985179631064430432954075411818769514_16) /r1) <= eps_q) result(1) = 1
  if (abs((r2 - 0.999767866461934619913139294582561925_16) /r2) <= eps_q) result(2) = 1
  
  call check(result, expect, 2)

end program test
