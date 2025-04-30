! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test dot_product with quad precision argument

program test
  use check_mod
  integer, parameter :: n = 6
  real(16) :: y1, y2, x1(2), x2(2)
  real(16) :: eps_q = 1.e-33_16
  integer :: result(n), expect(n), x1i(2), x2i(2), y1i, y2i
  complex :: x1c(2), x2c(2), yc(2), outp(2)

  expect = 1
  result = 0

  x1c(1)=(231.231, 234.43)
  x1c(2)=(-12.1, 1214.354)
  x2c(1)=(12.434, -90.545)
  x2c(2)=(-23.432, -20.76)
  yc(2) = dot_product(x1c, x2c)
  yc(1) = dot_product([(12.34,87.76),(45.98,76.76)],[(2321.09,43.65),(0.89,0.89)])
  outp(1) = (32582.2148,-203187.625)
  outp(2) = (-43277.7969,4854.22461)

  if(abs((yc(1)%im - outp(1)%im)/outp(1)%im).gt.5e-6) stop 1
  if(abs((yc(1)%re - outp(1)%re)/outp(1)%re).gt.5e-6) stop 2
  if(abs((yc(2)%im - outp(2)%im)/outp(2)%im).gt.5e-6) stop 3
  if(abs((yc(2)%re - outp(2)%re)/outp(2)%re).gt.5e-6) stop 4

  x1(1)=1.231_16
  x1(2)=213.21321_16
  x2(1)=2321.232143_16
  x2(2)=-0.31321303715884232312_16

  y1 = dot_product([-1.231_16,1231.231_16],[1.232143_16,1.232143_16])
  y2 = dot_product(x1, x2)
  if (abs((y1 - 1515.5358899999999999999999999999998464561134_16)/y1) <= eps_q) result(1) = 1
  if (abs((y2 - 2790.6556109665139484037275847999998671409724_16)/y2) <= eps_q) result(2) = 1

  x1(1)=1232.231_16
  x1(2)=2123.21321_16
  x2(1)=92321.232143_16
  x2(2)=-2130.3321321303715884232312_16
  y1 = dot_product([-121.231_16,121.231_16],[-1.243_16,1.232143_16])
  y2 = dot_product(x1, x2)
  if (abs((y1 - 300.0640610330000000000000000000000116645644_16)/y1) <= eps_q) result(3) = 1
  if (abs((y2 - 109237934.8801743626012511124452758491229459622198_16)/y2) <= eps_q) result(4) = 1

  x1i(1)=32_8
  x1i(2)=123_8
  x2i(1)=9321_8
  x2i(2)=-230_8
  y1i = dot_product([-121_8,121_8],[-1_8,132_8])
  y2i = dot_product(x1i, x2i)
  if (y1i == 16093_8) result(5) = 1
  if (y2i == 269982_8) result(6) = 1

  call check(result, expect, n)

end program test
