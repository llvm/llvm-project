! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for negative operation of quad complex.

program test
  use check_mod
  integer, parameter :: n = 12
  complex(16) :: c1, rst1, rst2
  complex(16) :: rst3 = - (15.213513214321532163123412431412417090_16, 9.1214985738130981578153850981181281123_16)
  complex(16), parameter :: rst4 = - (1.2135132143215321631234124311241709083_16, 2.1298419028097583091849810928910931241_16)
  complex(16), parameter :: rst5(2) = - [(1.0_16, 2.0_16), (-3.0_16, -4.0_16)]
  real(16) :: result(n), expect(n)

  c1 = (42.58879788522658767867845243213578_16, -7.3387123548975562114863159753156523_16)

  rst1 = - c1
  rst2 = - (-1.3789662336879942354856698413366981_16, 7.1233687476876737257454577868787564_16)

  expect(1) = -42.58879788522658767867845243213578_16
  expect(2) = 7.3387123548975562114863159753156523_16
  expect(3) = 1.3789662336879942354856698413366981_16
  expect(4) = -7.1233687476876737257454577868787564_16
  expect(5) = -15.213513214321532163123412431412417090_16
  expect(6) = -9.1214985738130981578153850981181281123_16
  expect(7) = -1.2135132143215321631234124311241709083_16
  expect(8) = -2.1298419028097583091849810928910931241_16
  expect(9) = -1.0_16
  expect(10) = -2.0_16
  expect(11) = 3.0_16
  expect(12) = 4.0_16

  result(1) = rst1%re
  result(2) = rst1%im
  result(3) = rst2%re
  result(4) = rst2%im
  result(5) = rst3%re
  result(6) = rst3%im
  result(7) = rst4%re
  result(8) = rst4%im
  result(9) = rst5(1)%re
  result(10) = rst5(1)%im
  result(11) = rst5(2)%re
  result(12) = rst5(2)%im

  call checkr16(result, expect, n)
end
