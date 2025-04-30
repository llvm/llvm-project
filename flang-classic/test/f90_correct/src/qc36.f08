! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for arbitrary constant word convert to complex(16)

program test
  integer, parameter :: n = 4
  integer, parameter :: m = n * 3
  complex(16) :: result(m), expect(m)
  complex(16), parameter :: rst1 = z'12'
  complex(16), parameter :: rst2 = z'1234'
  complex(16), parameter :: rst3 = z'123456'
  complex(16), parameter :: rst4 = z'12345678'
  complex(16) :: rst5 = z'12'
  complex(16) :: rst6 = z'1234'
  complex(16) :: rst7 = z'123456'
  complex(16) :: rst8 = z'12345678'

  expect(1) = (9.842266181545798168605147216506023E-4964_16, 0.0_16)
  expect(2) = (2.527260849116661200793808525396250E-4961_16, 0.0_16)
  expect(3) = (6.469842165209655953443081590301650E-4959_16, 0.0_16)
  expect(4) = (1.656279626669547521271554441739417E-4956_16, 0.0_16)
  expect(5) = (1.165531521498844519966399012480976E-4964_16, 0.0_16)
  expect(6) = (3.017431605658119701690788554534083E-4962_16, 0.0_16)
  expect(7) = (7.725181775545058106487958201357661E-4960_16, 0.0_16)
  expect(8) = (1.977647311560549207823930610480236E-4957_16, 0.0_16)
  expect(9) = (1.165531521498844519966399012480976E-4964_16, 0.0_16)
  expect(10) = (3.017431605658119701690788554534083E-4962_16, 0.0_16)
  expect(11) = (7.725181775545058106487958201357661E-4960_16, 0.0_16)
  expect(12) = (1.977647311560549207823930610480236E-4957_16, 0.0_16)

  result(1) = z'98'
  result(2) = z'9876'
  result(3) = z'987654'
  result(4) = z'98765432'
  result(n+1) = rst1
  result(n+2) = rst2
  result(n+3) = rst3
  result(n+4) = rst4
  result(n+5) = rst5
  result(n+6) = rst6
  result(n+7) = rst7
  result(n+8) = rst8

  do i = 1, m
    if (result(i) .ne. expect(i)) STOP i
  enddo

  print *, 'PASS'

end program
