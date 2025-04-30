! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for power take quad precision

program power
  parameter(n = 8)
  integer(4) :: i = 4
  integer(8) :: k = 8 
  real(16) :: eps_q = 1.e-33_16
  real(16) :: q1, q2, q3, q4, q5, q6
  real(16) :: qi, qk, qq1, qq2, qq3, qq4, qq5, qq6
  integer :: result(n), expect(n)
  q1 = 3.1415926_16
  q2 = 0.25_16
  q3 = 0.5_16
  q4 = 0.75_16
  q5 = 2.5_16
  q6 = 1.5_16
  qi = q1 ** i
  qk = q1 ** k
  qq1 = q1 ** q1
  qq2 = q1 ** q2
  qq3 = q1 ** q3
  qq4 = q1 ** q4
  qq5 = q1 ** q5
  qq6 = q1 ** q6
  expect = 1
  result = 0
  
  if ((abs((qi - 97.4090843875227817341897250576000083_16) / qi)) <= eps_q) result(1) = 1
  if ((abs((qk - 9488.52972121553454589261156314977847_16) / qk)) <= eps_q) result(2) = 1
  if ((abs((qq1 - 36.4621554164068422726887018513141888_16) / qq1)) <= eps_q) result(3) = 1
  if ((abs((qq2 - 1.33133535812285643584815073530608718_16) / qq2)) <= eps_q) result(4) = 1
  if ((abs((qq3 - 1.77245383578811439802009728948269621_16) / qq3)) <= eps_q) result(5) = 1
  if ((abs((qq4 - 2.35973046222519985520335189052097894_16) / qq4)) <= eps_q) result(6) = 1
  if ((abs((qq5 - 17.4934175816110073050960195137558137_16) / qq5)) <= eps_q) result(7) = 1
  if ((abs((qq6 - 5.56832785435355536077339229591889638_16) / qq6)) <= eps_q) result(8) = 1

  call check(result, expect, 8)

end program power
