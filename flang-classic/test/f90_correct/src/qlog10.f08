! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test LOG10 intrinsic with quad-precision argument

program test
  integer, parameter :: n = 5
  real(kind = 16) :: x, ylog10(4), answer
  real(kind = 16) :: eps_q = 1.e-33_16
  real(kind = 16), parameter :: r1 = log10(1.23456_16)
  integer :: real_result(n), exp_result(n)
  exp_result = 1
  real_result = 0

  ylog10(1) = 4932.0754489586679023818980511660936429069020_16
  ylog10(2) = 1.0504226808431285076551442458397724825252_16
  ylog10(3) = 1.0000000000000000000000000000000000000000_16
  ylog10(4) = 9.15122016277716810693997770679058130E-0002_16
  x = huge(x)
  answer = log10(x)
  if (abs((answer - ylog10(1))/answer) <= eps_q) real_result(1) = 1
  x = 11.23111_16
  answer = log10(x)
  if (abs((answer - ylog10(2))/answer) <= eps_q) real_result(2) = 1
  answer = log10(11.23111_16)
  if (abs((answer - ylog10(2))/answer) <= eps_q) real_result(3) = 1
  answer = log10(10.0_16)
  if (abs((answer - ylog10(3))/answer) <= eps_q) real_result(4) = 1
  if (abs((r1 - ylog10(4))/r1) <= eps_q) real_result(5) = 1

  call check(real_result, exp_result, n)

end program test
