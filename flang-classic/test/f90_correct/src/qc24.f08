! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for constant integer(8) convert to complex(16)

program test
  integer, parameter :: n = 5
  integer, parameter :: m = n * 4
  integer(8) :: i_8(n)
  complex(16) :: result(m), expect(m)
  complex(16), parameter :: rst1 = -huge(0_8) -1_8
  complex(16), parameter :: rst2 = -7_8
  complex(16), parameter :: rst3 = 0_8
  complex(16), parameter :: rst4 = 77_8
  complex(16), parameter :: rst5 = huge(0_8)
  complex(16) :: rst6 = -huge(0_8) -1_8
  complex(16) :: rst7 = -7_8
  complex(16) :: rst8 = 0_8
  complex(16) :: rst9 = 77_8
  complex(16) :: rst10 = huge(0_8)

  expect(1) = (-9223372036854775808.0_16, 0.0_16) ! (-2^63, 0)
  expect(2) = (-7.0_16, 0.0_16)
  expect(3) = (0.0_16, 0.0_16)
  expect(4) = (77.0_16, 0.0_16)
  expect(5) = (9223372036854775807.0_16, 0.0_16) ! (2^63 - 1, 0)
  expect(6) = (-9223372036854775808.0_16, 0.0_16) ! (-2^63, 0)
  expect(7) = (-7.0_16, 0.0_16)
  expect(8) = (0.0_16, 0.0_16)
  expect(9) = (77.0_16, 0.0_16)
  expect(10) = (9223372036854775807.0_16, 0.0_16) ! (2^63 - 1, 0)
  expect(11) = (-9223372036854775808.0_16, 0.0_16) ! (-2^63, 0)
  expect(12) = (-7.0_16, 0.0_16)
  expect(13) = (0.0_16, 0.0_16)
  expect(14) = (77.0_16, 0.0_16)
  expect(15) = (9223372036854775807.0_16, 0.0_16) ! (2^63 - 1, 0)
  expect(16) = (-9223372036854775808.0_16, 0.0_16) ! (-2^63, 0)
  expect(17) = (-7.0_16, 0.0_16)
  expect(18) = (0.0_16, 0.0_16)
  expect(19) = (77.0_16, 0.0_16)
  expect(20) = (9223372036854775807.0_16, 0.0_16) ! (2^63 - 1, 0)

  i_8(1) = -huge(0_8) -1_8
  i_8(2) = -7_8
  i_8(3) = 0_8
  i_8(4) = 77_8
  i_8(5) = huge(0_8)

  do i = 1, n
    result(i) = i_8(i)
  enddo

  result(n+1) = rst1
  result(n+2) = rst2
  result(n+3) = rst3
  result(n+4) = rst4
  result(n+5) = rst5
  result(n+6) = -huge(0_8) -1_8
  result(n+7) = -7_8
  result(n+8) = 0_8
  result(n+9) = 77_8
  result(n+10) = huge(0_8)
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
