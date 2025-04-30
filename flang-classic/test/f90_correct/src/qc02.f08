! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for exponentiation operation of real complex : complex(4) ^ complex(4)

program test
  integer, parameter :: n = 16
  integer, parameter :: m = 2 * n
  real(4), parameter :: f_tol = 5E-6_4
  integer :: i
  complex(4) :: c1, czero
  complex(4) :: rst1 = (0.544749_4, -0.334712_4) **&
                       (77.77777_4, 6.112434_4)
  complex(4) :: rst2 = (2.544744_4, -1.353156_4) **&
                       (-17.77777_4, -17.12442_4)
  complex(4) :: rst3 = (2.544474_4, -1.333156_4) **&
                       (0.0_4, 0.0_4)
  complex(4) :: rst4 = (2.563574_4, -1.334656_4) **&
                       (1.0_4, 1.0_4)
  complex(4) :: rst5 = (2.213574_4, -1.333156_4) **&
                       (2.0_4, 2.0_4)
  complex(4) :: rst6 = (0.0_4, 0.0_4) ** (0.0_4, 0.0_4)
  complex(4) :: rst7 = (0.0_4, 0.0_4) ** (1.0_4, 0.0_4)
  complex(4) :: rst8 = (0.0_4, 0.0_4) ** (2.0_4, 0.0_4)

  complex(4), parameter :: rst9 = (-15.54475_4, 33.45913_4) **&
                                  (7.77777_4, 2.15330_4)
  complex(4), parameter :: rst10 = (-223.5445_4, 133.1233_4) **&
                                   (-17.77777_4, -69.12439_4)
  complex(4), parameter :: rst11 = (1.544797_4, 2.334712_4) **&
                                   (0.0_4, 0.0_4)
  complex(4), parameter :: rst12 = (1.547974_4, 2.334144_4) **&
                                   (1.0_4, 1.0_4)
  complex(4), parameter :: rst13 = (1.544774_4, 2.334746_4) **&
                                   (2.0_4, 2.0_4)
  complex(4), parameter :: rst14 = (0.0_4, 0.0_4) ** (0.0_4, 0.0_4)
  complex(4), parameter :: rst15 = (0.0_4, 0.0_4) ** (1.0_4, 0.0_4)
  complex(4), parameter :: rst16 = (0.0_4, 0.0_4) ** (2.0_4, 0.0_4)
  real(4) :: result(2*n), expect(2*n)

  expect(1) = -7.61637468E-16_4
  expect(2) = -2.25817036E-14_4
  expect(3) = -1.55814619E-12_4
  expect(4) = 2.12768273E-14_4
  expect(5) = 1.0_4
  expect(6) = 0.0_4
  expect(7) = 3.90359426_4
  expect(8) = 2.56484795_4
  expect(9) = 13.5486765_4
  expect(10) = 14.3621616_4
  expect(11) = 1.0_4
  expect(12) = 0.0_4
  expect(13) = 0.0_4
  expect(14) = 0.0_4
  expect(15) = 0.0_4
  expect(16) = 0.0_4
  expect(17) = -3.92659789E+09_4
  expect(18) = -2.01190461E+10_4
  expect(19) = -1.67893647E+35_4
  expect(20) = 5.75790077E+34_4
  expect(21) = 1.0_4
  expect(22) = 0.0_4
  expect(23) = -0.449486375_4
  expect(24) = 0.944178343_4
  expect(25) = -0.686288536_4
  expect(26) = -0.847059488_4
  expect(27) = 1.0_4
  expect(28) = 0.0_4
  expect(29) = 0.0_4
  expect(30) = 0.0_4
  expect(31) = 0.0_4
  expect(32) = 0.0_4

  result(1) = rst1%re
  result(2) = rst1%im
  result(3) = rst2%re
  result(4) = rst2%im
  result(5) = rst3%re
  result(6) = rst3%im
  result(7) = rst4%re
  result(8) = rst4%im
  result(9) = rst5%re
  result(10) = rst5%im
  result(11) = rst6%re
  result(12) = rst6%im
  result(13) = rst7%re
  result(14) = rst7%im
  result(15) = rst8%re
  result(16) = rst8%im
  result(17) = rst9%re
  result(18) = rst9%im
  result(19) = rst10%re
  result(20) = rst10%im
  result(21) = rst11%re
  result(22) = rst11%im
  result(23) = rst12%re
  result(24) = rst12%im
  result(25) = rst13%re
  result(26) = rst13%im
  result(27) = rst14%re
  result(28) = rst14%im
  result(29) = rst15%re
  result(30) = rst15%im
  result(31) = rst16%re
  result(32) = rst16%im

  do i = 1, m
    if(expect(i) .eq. 0.0_4) then
      if (result(i) .ne. expect(i)) STOP i
    else
      if (abs((result(i) - expect(i)) / expect(i)) .gt. f_tol) STOP i
    endif
  enddo

  print *, 'PASS'
end
