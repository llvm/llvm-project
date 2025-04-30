! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for constant integer(2) convert to complex(16)

program test
  integer, parameter :: n = 5
  integer, parameter :: m = n * 4
  integer(2) :: i_2(n)
  complex(16) :: result(m), expect(m)
  complex(16), parameter :: rst1 = -huge(0_2) -1_2
  complex(16), parameter :: rst2 = -7_2
  complex(16), parameter :: rst3 = 0_2
  complex(16), parameter :: rst4 = 77_2
  complex(16), parameter :: rst5 = huge(0_2)
  complex(16) :: rst6 = -huge(0_2) -1_2
  complex(16) :: rst7 = -7_2
  complex(16) :: rst8 = 0_2
  complex(16) :: rst9 = 77_2
  complex(16) :: rst10 = huge(0_2)

  expect(1) = (-32768.0_16, 0.0_16)
  expect(2) = (-7.0_16, 0.0_16)
  expect(3) = (0.0_16, 0.0_16)
  expect(4) = (77.0_16, 0.0_16)
  expect(5) = (32767.0_16, 0.0_16)
  expect(6) = (-32768.0_16, 0.0_16)
  expect(7) = (-7.0_16, 0.0_16)
  expect(8) = (0.0_16, 0.0_16)
  expect(9) = (77.0_16, 0.0_16)
  expect(10) = (32767.0_16, 0.0_16)
  expect(11) = (-32768.0_16, 0.0_16)
  expect(12) = (-7.0_16, 0.0_16)
  expect(13) = (0.0_16, 0.0_16)
  expect(14) = (77.0_16, 0.0_16)
  expect(15) = (32767.0_16, 0.0_16)
  expect(16) = (-32768.0_16, 0.0_16)
  expect(17) = (-7.0_16, 0.0_16)
  expect(18) = (0.0_16, 0.0_16)
  expect(19) = (77.0_16, 0.0_16)
  expect(20) = (32767.0_16, 0.0_16)

  i_2(1) = -huge(0_2) -1_2
  i_2(2) = -7_2
  i_2(3) = 0_2
  i_2(4) = 77_2
  i_2(5) = huge(0_2)

  do i = 1, n
    result(i) = i_2(i)
  enddo

  result(n+1) = rst1
  result(n+2) = rst2
  result(n+3) = rst3
  result(n+4) = rst4
  result(n+5) = rst5
  result(n+6) = -huge(0_2) -1_2
  result(n+7) = -7_2
  result(n+8) = 0_2
  result(n+9) = 77_2
  result(n+10) = huge(0_2)
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
