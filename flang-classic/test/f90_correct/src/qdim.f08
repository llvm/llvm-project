! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test DIM intrinsic with quad-precision arguments

program test
  real(kind = 16) :: x1, x2, ydim(3), answer
  real(kind = 16) :: eps_q = 1.e-33_16
  integer :: real_result(6), exp_result(6) 
  exp_result = 1
  real_result = 0
 
  x1 = 1670.9890_16
  x2 = 989.785569_16
  ydim(1) = 0.0999999999999999999999999999999996918512_16
  ydim(2) = 0.0000000000000000000000000000000000000000_16
  ydim(3) = 681.2034309999999999999999999999999250140378_16
  answer = dim(x1, x2)
  if (abs((answer - ydim(3))/answer) <= eps_q) real_result(1) = 1
  if (abs(dim(x2, x1) - ydim(2)) <= eps_q) real_result(2) = 1
  answer = dim(10.0_16, 9.9_16)
  if (abs((answer - ydim(1))/answer) <= eps_q) real_result(3) = 1
  if (abs(dim(9.9_16, 10.0_16) - ydim(2)) <= eps_q) real_result(4) = 1
  if (abs(dim(9.9_16, 9.9_16) - ydim(2)) <= eps_q) real_result(5) = 1
  if (abs(dim(x1, x1) - ydim(2)) <= eps_q) real_result(6) = 1

  call check(real_result, exp_result, 6)

end program test
