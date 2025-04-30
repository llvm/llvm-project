! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for constant integer(4) convert to complex(16)

program test
  integer, parameter :: n = 5
  integer, parameter :: m = n * 4
  integer(4) :: i_4(n)
  complex(16) :: result(m), expect(m)
  complex(16), parameter :: rst1 = -huge(0_4) -1_4
  complex(16), parameter :: rst2 = -7_4
  complex(16), parameter :: rst3 = 0_4
  complex(16), parameter :: rst4 = 77_4
  complex(16), parameter :: rst5 = huge(0_4)
  complex(16) :: rst6 = -huge(0_4) -1_4
  complex(16) :: rst7 = -7_4
  complex(16) :: rst8 = 0_4
  complex(16) :: rst9 = 77_4
  complex(16) :: rst10 = huge(0_4)

  expect(1) = (-2147483648.0_16, 0.0_16) ! (-2^31, 0)
  expect(2) = (-7.0_16, 0.0_16)
  expect(3) = (0.0_16, 0.0_16)
  expect(4) = (77.0_16, 0.0_16)
  expect(5) = (2147483647.0_16, 0.0_16) ! (2^31 - 1, 0)
  expect(6) = (-2147483648.0_16, 0.0_16) ! (-2^31, 0)
  expect(7) = (-7.0_16, 0.0_16)
  expect(8) = (0.0_16, 0.0_16)
  expect(9) = (77.0_16, 0.0_16)
  expect(10) = (2147483647.0_16, 0.0_16) ! (2^31 - 1, 0)
  expect(11) = (-2147483648.0_16, 0.0_16) ! (-2^31, 0)
  expect(12) = (-7.0_16, 0.0_16)
  expect(13) = (0.0_16, 0.0_16)
  expect(14) = (77.0_16, 0.0_16)
  expect(15) = (2147483647.0_16, 0.0_16) ! (2^31 - 1, 0)
  expect(16) = (-2147483648.0_16, 0.0_16) ! (-2^31, 0)
  expect(17) = (-7.0_16, 0.0_16)
  expect(18) = (0.0_16, 0.0_16)
  expect(19) = (77.0_16, 0.0_16)
  expect(20) = (2147483647.0_16, 0.0_16) ! (2^31 - 1, 0)

  i_4(1) = -huge(0_4) -1_4
  i_4(2) = -7_4
  i_4(3) = 0_4
  i_4(4) = 77_4
  i_4(5) = huge(0_4)

  do i = 1, n
    result(i) = i_4(i)
  enddo

  result(n+1) = rst1
  result(n+2) = rst2
  result(n+3) = rst3
  result(n+4) = rst4
  result(n+5) = rst5
  result(n+6) = -huge(0_4) -1_4
  result(n+7) = -7_4
  result(n+8) = 0_4
  result(n+9) = 77_4
  result(n+10) = huge(0_4)
  result(n+11) = rst6
  result(n+12) = rst7
  result(n+13) = rst8
  result(n+14) = rst9
  result(n+15) = rst10

  do i = 1, m
    if (result(i) .ne. expect(i)) STOP i
  enddo

  print *, 'PASS'

end program
