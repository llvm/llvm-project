! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for comparison operation of quad complex.

program test
  integer, parameter :: n = 7
  complex(16) :: c1, c2, czero
  logical :: rst1, rst2, rst3, rst4, rst5
  logical :: rst6 = (1.23456789012345678901234567890123456789_16, 9.876543210987654321098765431098765431_16) /=&
                    (1.23456789012345678901234567880123456789_16, 9.876543210987654321098765431098765431_16)
  logical, parameter :: rst7 = (-1.23456789012345678901234567890123456789_16, -9.876543210987654321098765431098765431_16) /=&
                               (-1.23456789012345678901234567890123456789_16, -9.876543210987654321098765431088765431_16)
  logical :: result(n), expect(n)
  c1 = (42.58879788522658767867845243213578_16, -7.3387123548975562114863159753156_16)
  c2 = (-13.78966233687994235485669841336698_16, 71.233687476876737257454577868787_16)
  czero = (+0.0_16, -0.0_16)

  rst1 = c1 == c2
  rst2 = c1 /= c2
  rst3 = c1 == (42.58879788522658767867845243213578_16, -7.3387123548975562114863159753156_16)
  rst4 = c1 /= (-1.378966233687994235485669841336698_16, 7.1233687476876737257454577868787_16)
  rst5 = czero == (0.0_16, 0.0_16)

  expect = [.false., .true., .true., .true., .true., .true., .true.]

  result(1) = rst1
  result(2) = rst2
  result(3) = rst3
  result(4) = rst4
  result(5) = rst5
  result(6) = rst6
  result(7) = rst7

  call check(result, expect, n)
end
