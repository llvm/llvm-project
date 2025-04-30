! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test SINH/COSH/TANH intrinsics with quad-precision arguments

program test
  real(kind = 16) :: x, ysinh, ycosh, ytanh, answer
  real(kind = 16) :: eps_q = 1.e-33_16
  integer :: real_result(9), exp_result(9)
  exp_result = 1
  real_result = 0

  ysinh = 3.23716527388468658691437160346409490_16
  ycosh = 3.38810256787555325082227311755750319_16
  ytanh = 0.955450789647873886823057256870080932_16
  answer = sinh(1.8908908_16)
  if (abs((answer - ysinh)/answer) <= eps_q) real_result(1) = 1
  answer = cosh(1.8908908_16)
  if (abs((answer - ycosh)/answer) <= eps_q) real_result(2) = 1
  answer = tanh(1.8908908_16)
  if (abs((answer - ytanh)/answer) <= eps_q) real_result(3) = 1

  x = 0.0000_16
  ysinh = 0.00000000000000000000000000000000000_16
  ycosh = 1.00000000000000000000000000000000000_16
  ytanh = 0.00000000000000000000000000000000000_16
  answer = sinh(x)
  if (abs(answer - ysinh) <= eps_q) real_result(4) = 1
  answer = cosh(x)
  if (abs((answer - ycosh)/answer) <= eps_q) real_result(5) = 1
  answer = tanh(x)
  if (abs(answer - ytanh) <= eps_q) real_result(6) = 1

  x = 8.8989_16
  ysinh = 3661.95633297092335408378314352915929_16
  ycosh = 3661.95646950996020078989751571492122_16
  ytanh = 0.999999962714183529063727950597920453_16
  answer = sinh(x)
  if (abs(answer - ysinh) <= eps_q) real_result(7) = 1
  answer = cosh(x)
  if (abs(answer - ycosh) <= eps_q) real_result(8) = 1
  answer = tanh(x)
  if (abs(answer - ytanh) <= eps_q) real_result(9) = 1

  call check(real_result, exp_result, 9)

end program test
