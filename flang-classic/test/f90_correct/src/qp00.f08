! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for negative operation of quad precision.

program test
  use check_mod
  integer, parameter :: n = 6
  real(16) :: c1, rst1, rst2
  real(16) :: rst3 = - (-15.213513214321532163123412431412417090_16)
  real(16), parameter :: rst4 = - (-1.2135132143215321631234124311241709083_16)
  real(16), parameter :: rst5(2) = - [1.0_16, -3.0_16]
  real(16) :: result(n), expect(n)

  c1 = 42.58879788522658767867845243213578_16

  rst1 = - c1
  rst2 = -(-7.1233687476876737257454577868787564_16)

  expect(1) = -42.58879788522658767867845243213578_16
  expect(2) = 7.1233687476876737257454577868787564_16
  expect(3) = 15.213513214321532163123412431412417090_16
  expect(4) = 1.2135132143215321631234124311241709083_16
  expect(5) = -1.0_16
  expect(6) = 3.0_16

  result(1) = rst1
  result(2) = rst2
  result(3) = rst3
  result(4) = rst4
  result(5) = rst5(1)
  result(6) = rst5(2)

  call checkr16(result, expect, n)
end
