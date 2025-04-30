! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test I/O of quad-precision values

program test
  integer, parameter :: k = 16, n = 1
  real(kind = k) :: tmpb(3) = [1111.123456789123456789123456789123456789_16, 1.22222222222222_16, 1.33333333333333_16]
  character(240) :: str
  integer :: result(n), expect(n)
  expect = 1
  result = 0

  write(str,100) tmpb(1:3)
  100 FORMAT('',e45.36,5X,e45.36,5X,e45.36)
  if(str == "   0.111112345678912345678912345678912345E+04        0.122222222222221999999999999999999997E+01        0.133333333333332999999999999999999996E+01") result(1) = 1
  call check(result, expect, n)

end program
