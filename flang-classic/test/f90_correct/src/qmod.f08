! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test MOD intrinsic with quad-precision arguments

program p
  use check_mod
  integer, parameter :: n = 4
  integer, parameter :: k = 16
  real(16) :: rslts(n), expect(n), r = mod(1.000002_16,0.9999999999_16), er
  real(kind = k) :: t1, t2

  expect(1) = 28531.9497157199999999999999684411083_16
  expect(2) = 0.648722219999999999999940119212164753_16
  expect(3) = 1.92345000000000000000000000000000007_16
  expect(4) = 123465.455999999999999999999999999997_16
  er = 2.00009999999999999999999999988962226E-0006_16

  t1 = 1111111111111.92222222_16
  rslts(1) = mod(t1, 545465.7878787_16)
  t2 = 1.92345_16
  rslts(2) = mod(t1, t2)
  rslts(3) = mod(t2, t1)
  rslts(4) = mod(123465.456_16, 145678.12345_16)
  if(abs((r - er) / r) > 1e-33_16) stop 1

  call checkr16(rslts, expect, n)

end program p
