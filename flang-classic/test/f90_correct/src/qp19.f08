! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test I/O of quad-precision values

program test
  integer, parameter :: k = 16, n = 1
  real(kind = k) :: tmpa = 0.000123456789123456789123456789_16
  character(80) :: str1
  integer :: result(n), expect(n)
  expect = 1
  result = 0

  write(str1,100) tmpa
  100 FORMAT('',F)
  if (str1 == "       0.000123456789123456789123456789000000") result(1) = 1
  call check(result, expect, n)

end program
