! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test scaled complementary error function intrinsic (ERFC_SCALED) with quad-precision arguments

program p
  use ieee_arithmetic
  real(kind=16), parameter :: xneg = -106.5637380121098417363881585073946045229689_16
  real(kind=16), parameter :: sqrpi = 5.6418958354775628694807945156077263153528602528974e-1_16
  real(kind=16), parameter :: xchg = 12.0_16
  real(kind=16), parameter :: xsmall = 1.0e-20_16
  integer, parameter :: n = 11
  real(kind=16), dimension(n) :: X, result, expect
  integer i

  X(1) = xneg + xsmall
  X(2) = 0.0_16
  X(3) = 1.0_16
  X(4) = xchg - xsmall
  X(5) = xchg
  X(6) = xchg + xsmall
  X(7) = huge(1.0_16) * sqrpi
  X(8) = huge(1.0_16) - xsmall
  X(9) = huge(1.0_16)
  X(10) = xneg
  X(11) = xneg - xsmall

  expect(1) = 1.18973149535723176255011461911045627E+4932_16
  expect(2) = 1.00000000000000000000000000000000000_16
  expect(3) = 0.427583576155807004410750344490515140_16
  expect(4) = 4.68542210148937626196271920267778740E-0002_16
  expect(5) = 4.68542210148937626195884133993966638E-0002_16
  expect(6) = 4.68542210148937626195496347720160375E-0002_16
  expect(7) = 8.40525785778023376565669454330438151E-4933_16
  expect(8) = 4.74215893039253622890723952726434009E-4933_16
  expect(9) = 4.74215893039253622890723952726434009E-4933_16

  do i=1, n
    result(i) = erfc_scaled(X(i))
    if (i >= 10) then
      if (ieee_is_finite(result(i))) STOP i
    endif
    if ((i < 10) .and. (abs(result(i) - expect(i)) / expect(i) > 1.e-33_16)) STOP i
  end do

  print *, 'PASS'
end program p
