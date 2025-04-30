! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test ATAND intrinsic with quad-precision arguments

program test
  real(16) :: r1, r2, q1 = -0.123456789_16
  real(16) :: eps_q = 1.e-33_16
  integer :: result(2), expect(2)
  expect = 1
  result = 0

  r1 = atand(0.987654321_16)
  r2 = atand(q1)

  if (abs((r1 - 44.6441301697361925144580643835703598_16)/r1) <= eps_q) result(1) = 1
  if (abs((r2 - (-7.03794069978169795284658606936286925_16))/r2) <= eps_q) result(2) = 1

  call check(result, expect, 2)

end program test
