! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test I/O of quad-precision values

program test
  integer, parameter :: k = 16, n = 2
  real(kind = k) :: tmpa = -0.000123456789123456789123456789123456789_16
  real(kind = k) :: tmpb = 0.000123456789123456789123456789123456789_16
  character(80) :: str1, str2
  integer :: result(n), expect(n)
  expect = 1
  result = 0

  write(str1,100) tmpa
  write(str2,100) tmpb
  100 FORMAT('',2pG0.28,5X,-2pG0.28)
  if(str1 == "-12.345678912345678912345678912E-05") result(1) = 1
  if(str2 == "12.345678912345678912345678912E-05") result(2) = 1
  call check(result, expect, n)

end program
