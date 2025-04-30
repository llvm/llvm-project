! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test MAX/MIN intrinsics with quad-precision arguments

program p
  use check_mod
  integer, parameter :: n = 8
  integer, parameter :: k = 16
  real(kind = k), parameter :: x1 = 1.2345678123151412321321541_16
  real(kind = k), parameter :: x2 = 1.5212346451341243213124125_16
  real(kind = k), parameter :: x3 = 1.5212345451341221215321532_16
  real(kind = k), parameter :: x4 = max(x1, x2, x3)
  real(kind = k), parameter :: x5 = min(x1, x2, x3)
  real(kind = k) :: x6 = MAX(1.5212346451341243213124125_16, 1.5212345451341221215321532_16)
  real(kind = k) :: x7 = MIN(1.5212346451341243213124125_16, 1.5212345451341221215321532_16)
  real(kind = k) :: rslts(n), expect(n)

  expect(1) = 23063235587371561727661983816373740.0_16
  expect(2) = -1.23450031356120897644896330052846060E-0003_16
  expect(3) = 1.18973149535723176508575932662800702E+4932_16
  expect(4) = -99999999999999.9000000000000000000054_16
  expect(5) = 1.5212346451341243213124125_16
  expect(6) = 1.2345678123151412321321541_16
  expect(7) = 1.5212346451341243213124125_16
  expect(8) = 1.5212345451341221215321532_16

  rslts(1) = max(-1.23450031356120897644896330052846060E-0003_16 , &
                 -0.00000000000000000000000000000000000_16 ,       &
                 74773.7620867837525752615166950535108_16 ,        &
                 23063235587371561727661983816373740.0_16 ,        &
                 6.93147180559945309417232121458176537_16 ,        &
                 99999999999999999999999999.9999999999_16)
  rslts(2) = min(-1.23450031356120897644896330052846060E-0003_16 , &
                 -0.00000000000000000000000000000000000_16 ,       &
                 74773.7620867837525752615166950535108_16 ,        &
                 23063235587371561727661983816373740.0_16 ,        &
                 6.93147180559945309417232121458176537_16 ,        &
                 99999999999999999999999999.9999999999_16)
  rslts(3) = max(huge(1.0_16),999999999999999.9_16)
  rslts(4) = min(tiny(1.0_16),-99999999999999.9_16)
  rslts(5) = x4
  rslts(6) = x5
  rslts(7) = x6
  rslts(8) = x7

  call checkr16(rslts, expect, n)

end program p
