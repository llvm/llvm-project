! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for constant complex(8) convert to real(16)

program test
  integer, parameter :: n = 9
  integer, parameter :: m = n * 3
  real(16), parameter :: q_tol = 5E-33
  complex(8) :: c_8(n)
  real(16) :: result(m), expect(m)
  real(16), parameter :: rst1 = cmplx(-huge(0.0_8), -huge(0.0_8), kind = 8)
  real(16), parameter :: rst2 = cmplx(-tiny(0.0_8), -tiny(0.0_8), kind = 8)
  real(16), parameter :: rst3 = cmplx(-epsilon(0.0_8), -epsilon(0.0_8), kind = 8)
  real(16), parameter :: rst4 = (-7.1234567890123456789_8, -7.1234567890123456789_8)
  real(16), parameter :: rst5 = (0.0_8, 0.0_8)
  real(16), parameter :: rst6 = (77.1234567890123456789_8, 77.1234567890123456789_8)
  real(16), parameter :: rst7 = cmplx(tiny(0.0_8), tiny(0.0_8), kind = 8)
  real(16), parameter :: rst8 = cmplx(epsilon(0.0_8), epsilon(0.0_8), kind = 8)
  real(16), parameter :: rst9 = cmplx(huge(0.0_8), huge(0.0_8), kind = 8)

  expect(1) = -1.797693134862315708145274237317043568E+0308_16
  expect(2) = -2.225073858507201383090232717332404064E-0308_16
  expect(3) = -2.220446049250313080847263336181640625E-0016_16
  expect(4) = -7.12345678901234524715846418985165656_16
  expect(5) = 0.0_16
  expect(6) = 77.1234567890123514644074020907282829_16
  expect(7) = 2.225073858507201383090232717332404064E-0308_16
  expect(8) = 2.220446049250313080847263336181640625E-0016_16
  expect(9) = 1.797693134862315708145274237317043568E+0308_16
  expect(10) = -1.797693134862315708145274237317043568E+0308_16
  expect(11) = -2.225073858507201383090232717332404064E-0308_16
  expect(12) = -2.220446049250313080847263336181640625E-0016_16
  expect(13) = -7.12345678901234524715846418985165656_16
  expect(14) = 0.0_16
  expect(15) = 77.1234567890123514644074020907282829_16
  expect(16) = 2.225073858507201383090232717332404064E-0308_16
  expect(17) = 2.220446049250313080847263336181640625E-0016_16
  expect(18) = 1.797693134862315708145274237317043568E+0308_16
  expect(19) = -1.797693134862315708145274237317043568E+0308_16
  expect(20) = -2.225073858507201383090232717332404064E-0308_16
  expect(21) = -2.220446049250313080847263336181640625E-0016_16
  expect(22) = -7.12345678901234524715846418985165656_16
  expect(23) = 0.0_16
  expect(24) = 77.1234567890123514644074020907282829_16
  expect(25) = 2.225073858507201383090232717332404064E-0308_16
  expect(26) = 2.220446049250313080847263336181640625E-0016_16
  expect(27) = 1.797693134862315708145274237317043568E+0308_16

  c_8(1) = cmplx(-huge(0.0_8), -huge(0.0_8), kind = 8)
  c_8(2) = cmplx(-tiny(0.0_8), -tiny(0.0_8), kind = 8)
  c_8(3) = cmplx(-epsilon(0.0_8), -epsilon(0.0_8), kind = 8)
  c_8(4) = (-7.1234567890123456789_8, -7.1234567890123456789_8)
  c_8(5) = (0.0_8, 0.0_8)
  c_8(6) = (77.1234567890123456789_8, 77.1234567890123456789_8)
  c_8(7) = cmplx(tiny(0.0_8), tiny(0.0_8), kind = 8)
  c_8(8) = cmplx(epsilon(0.0_8), epsilon(0.0_8), kind = 8)
  c_8(9) = cmplx(huge(0.0_8), huge(0.0_8), kind = 8)

  do i = 1, n
    result(i) = c_8(i)
  enddo

  result(n+1) = rst1
  result(n+2) = rst2
  result(n+3) = rst3
  result(n+4) = rst4
  result(n+5) = rst5
  result(n+6) = rst6
  result(n+7) = rst7
  result(n+8) = rst8
  result(n+9) = rst9
  result(n+10) = cmplx(-huge(0.0_8), -huge(0.0_8), kind = 8)
  result(n+11) = cmplx(-tiny(0.0_8), -tiny(0.0_8), kind = 8)
  result(n+12) = cmplx(-epsilon(0.0_8), -epsilon(0.0_8), kind = 8)
  result(n+13) = (-7.1234567890123456789_8, -7.1234567890123456789_8)
  result(n+14) = (0.0_8, 0.0_8)
  result(n+15) = (77.1234567890123456789_8, 77.1234567890123456789_8)
  result(n+16) = cmplx(tiny(0.0_8), tiny(0.0_8), kind = 8)
  result(n+17) = cmplx(epsilon(0.0_8), epsilon(0.0_8), kind = 8)
  result(n+18) = cmplx(huge(0.0_8), huge(0.0_8), kind = 8)

  do i = 1, m
    if (expect(i) .eq. 0.0_16) then
      if (result(i) .ne. expect(i)) STOP i
    else
      if (abs((result(i) - expect(i)) / expect(i)) .gt. q_tol) STOP i
    endif
  enddo

  print *, 'PASS'

end program
