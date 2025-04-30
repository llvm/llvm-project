! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for subtraction operation take quad precision

program test
  integer, parameter :: k = 16, n = 4
  real(kind = k) :: a1 = 1.191127656_16
  real(kind = k) :: a2 = 2.192357256_16
  real(kind = k) :: r1, r2, r3
  real(kind = k) :: r4 = 1.1_16 - 1.2_16
  real(kind = k) :: eps_q = 1.e-33_16

  integer :: result(n), expect(n)
  expect = 1
  result = 0

  r1 = a1 - a2
  r2 = a1 - 1.12345678_16
  r3 = 123.456789_16 + 987654.3125121651_16
  if (abs((r1 - (-1.00122959999999999999999999999999999_16))/r1) <= eps_q) result(1) = 1
  if (abs((r2 - 6.76708759999999999999999999999999435E-0002_16)/r2) <= eps_q) result(2) = 1
  if (abs((r3 - 987777.769301165100000000000000000006_16)/r3) <= eps_q) result(3) = 1
  if (abs((r4 - (-9.99999999999999999999999999999998844E-0002_16))/r4) <= eps_q) result(4) = 1
  call check(result, expect, n)

end program
