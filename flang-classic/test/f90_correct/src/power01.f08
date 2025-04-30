! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for power take quad precision

program power
  parameter(n = 5)
  real(16) :: qic = 2.0_16 ** 2 
  real(16) :: qrc = 2.0_16 ** 2.0_4
  real(16) :: qdc = 2.0_16 ** 2.0_8
  real(16) :: qqc = 2.0_16 ** 2.0_16, qqc2
  real(16) :: eps_q = 1.e-33_16
  integer :: result(n), expect(n)
  expect = 1
  result = 0
  qqc2 = 3.14_16 ** 4.13_16
  if (abs((qic - 4) / qic) <= eps_q) result(1) = 1
  if (abs((qrc - 4) / qrc) <= eps_q) result(2) = 1
  if (abs((qdc - 4) / qdc) <= eps_q) result(3) = 1
  if (abs((qqc - 4) / qqc) <= eps_q) result(4) = 1
  if (abs((qqc2 - 112.802687447605391549656731003386601_16) / qqc2) <= eps_q) result(5) = 1
  
  call check(result, expect, n)

end program power
