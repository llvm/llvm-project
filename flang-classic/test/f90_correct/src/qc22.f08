! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for constant real(8) convert to complex(16)

program test
  integer, parameter :: n = 9
  integer, parameter :: m = n * 4
  real(8), parameter :: d_tol = 5E-15
  real(8) :: r_8(n)
  complex(16) :: result(m), expect(m)
  complex(16), parameter :: rst1 = -huge(0.0_8)
  complex(16), parameter :: rst2 = -tiny(0.0_8)
  complex(16), parameter :: rst3 = -epsilon(0.0_8)
  complex(16), parameter :: rst4 = -7.1234567890123456789_8
  complex(16), parameter :: rst5 = 0.0_8
  complex(16), parameter :: rst6 = 77.1234567890123456789_8
  complex(16), parameter :: rst7 = tiny(0.0_8)
  complex(16), parameter :: rst8 = epsilon(0.0_8)
  complex(16), parameter :: rst9 = huge(0.0_8)
  complex(16) :: rst10 = -huge(0.0_8)
  complex(16) :: rst11 = -tiny(0.0_8)
  complex(16) :: rst12 = -epsilon(0.0_8)
  complex(16) :: rst13 = -7.1234567890123456789_8
  complex(16) :: rst14 = 0.0_8
  complex(16) :: rst15 = 77.1234567890123456789_8
  complex(16) :: rst16 = tiny(0.0_8)
  complex(16) :: rst17 = epsilon(0.0_8)
  complex(16) :: rst18 = huge(0.0_8)

  expect(1) = (-1.797693134862315708145274237317043568E+0308_16, 0.0_16)  ! (-real8_max, 0)
  expect(2) = (-2.225073858507201383090232717332404064E-0308_16, 0.0_16)  ! (-real8_min, 0)
  expect(3) = (-2.220446049250313080847263336181640625E-0016_16, 0.0_16)  ! (-real8_eps, 0)
  expect(4) = (-7.12345678901234524715846418985165656_16, 0.0_16)
  expect(5) = (0.0_16, 0.0_16)
  expect(6) = (77.1234567890123514644074020907282829_16, 0.0_16)
  expect(7) = (2.225073858507201383090232717332404064E-0308_16, 0.0_16)   ! (real8_min, 0)
  expect(8) = (2.220446049250313080847263336181640625E-0016_16, 0.0_16)   ! (real8_eps, 0)
  expect(9) = (1.797693134862315708145274237317043568E+0308_16, 0.0_16)   ! (real8_max, 0)
  expect(10) = (-1.797693134862315708145274237317043568E+0308_16, 0.0_16) ! (-real8_max, 0)
  expect(11) = (-2.225073858507201383090232717332404064E-0308_16, 0.0_16) ! (-real8_min, 0)
  expect(12) = (-2.220446049250313080847263336181640625E-0016_16, 0.0_16) ! (-real8_eps, 0)
  expect(13) = (-7.12345678901234524715846418985165656_16, 0.0_16)
  expect(14) = (0.0_16, 0.0_16)
  expect(15) = (77.1234567890123514644074020907282829_16, 0.0_16)
  expect(16) = (2.225073858507201383090232717332404064E-0308_16, 0.0_16)  ! (real8_min, 0)
  expect(17) = (2.220446049250313080847263336181640625E-0016_16, 0.0_16)  ! (real8_eps, 0)
  expect(18) = (1.797693134862315708145274237317043568E+0308_16, 0.0_16)  ! (real8_max, 0)
  expect(19) = (-1.797693134862315708145274237317043568E+0308_16, 0.0_16) ! (-real8_max, 0)
  expect(20) = (-2.225073858507201383090232717332404064E-0308_16, 0.0_16) ! (-real8_min, 0)
  expect(21) = (-2.220446049250313080847263336181640625E-0016_16, 0.0_16) ! (-real8_eps, 0)
  expect(22) = (-7.12345678901234524715846418985165656_16, 0.0_16)
  expect(23) = (0.0_16, 0.0_16)
  expect(24) = (77.1234567890123514644074020907282829_16, 0.0_16)
  expect(25) = (2.225073858507201383090232717332404064E-0308_16, 0.0_16)  ! (real8_min, 0)
  expect(26) = (2.220446049250313080847263336181640625E-0016_16, 0.0_16)  ! (real8_eps, 0)
  expect(27) = (1.797693134862315708145274237317043568E+0308_16, 0.0_16)  ! (real8_max, 0)
  expect(28) = (-1.797693134862315708145274237317043568E+0308_16, 0.0_16) ! (-real8_max, 0)
  expect(29) = (-2.225073858507201383090232717332404064E-0308_16, 0.0_16) ! (-real8_min, 0)
  expect(30) = (-2.220446049250313080847263336181640625E-0016_16, 0.0_16) ! (-real8_eps, 0)
  expect(31) = (-7.12345678901234524715846418985165656_16, 0.0_16)
  expect(32) = (0.0_16, 0.0_16)
  expect(33) = (77.1234567890123514644074020907282829_16, 0.0_16)
  expect(34) = (2.225073858507201383090232717332404064E-0308_16, 0.0_16)  ! (real8_min, 0)
  expect(35) = (2.220446049250313080847263336181640625E-0016_16, 0.0_16)  ! (real8_eps, 0)
  expect(36) = (1.797693134862315708145274237317043568E+0308_16, 0.0_16)  ! (real8_max, 0)

  r_8(1) = -huge(0.0_8)
  r_8(2) = -tiny(0.0_8)
  r_8(3) = -epsilon(0.0_8)
  r_8(4) = -7.1234567890123456789_8
  r_8(5) = 0.0_8
  r_8(6) = 77.1234567890123456789_8
  r_8(7) = tiny(0.0_8)
  r_8(8) = epsilon(0.0_8)
  r_8(9) = huge(0.0_8)

  do i = 1, n
    result(i) = r_8(i)
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
  result(n+10) = -huge(0.0_8)
  result(n+11) = -tiny(0.0_8)
  result(n+12) = -epsilon(0.0_8)
  result(n+13) = -7.1234567890123456789_8
  result(n+14) = 0.0_8
  result(n+15) = 77.1234567890123456789_8
  result(n+16) = tiny(0.0_8)
  result(n+17) = epsilon(0.0_8)
  result(n+18) = huge(0.0_8)
  result(n+19) = rst10
  result(n+20) = rst11
  result(n+21) = rst12
  result(n+22) = rst13
  result(n+23) = rst14
  result(n+24) = rst15
  result(n+25) = rst16
  result(n+26) = rst17
  result(n+27) = rst18

  do i = 1, m
    if (result(i)%im .ne. 0.0_16) STOP i
    if (expect(i)%re .eq. 0.0_16) then
      if (result(i)%re .ne. expect(i)%re) STOP i
    else
      if (abs((result(i)%re - expect(i)%re) / expect(i)%re) .gt. d_tol) STOP i
    endif
  enddo

  print *, 'PASS'

end program
