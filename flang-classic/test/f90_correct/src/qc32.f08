! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for constant complex(16) convert to real(16)

program test
  integer, parameter :: n = 9
  integer, parameter :: m = n * 4
  real(16), parameter :: q_tol = 5E-33
  complex(16) :: c_16(n)
  real(16) :: result(m), expect(m)
  real(16), parameter :: rst1 = cmplx(-huge(0.0_16), -huge(0.0_16), kind = 16)
  real(16), parameter :: rst2 = cmplx(-tiny(0.0_16), -tiny(0.0_16), kind = 16)
  real(16), parameter :: rst3 = cmplx(-epsilon(0.0_16), -epsilon(0.0_16), kind = 16)
  real(16), parameter :: rst4 = (-7.123456789012345678901234567890123456789_16,&
                                   -7.123456789012345678901234567890123456789_16)
  real(16), parameter :: rst5 = (0.0_16, 0.0_16)
  real(16), parameter :: rst6 = (77.123456789012345678901234567890123456789_16,&
                                   77.123456789012345678901234567890123456789_16)
  real(16), parameter :: rst7 = cmplx(tiny(0.0_16), tiny(0.0_16), kind = 16)
  real(16), parameter :: rst8 = cmplx(epsilon(0.0_16), epsilon(0.0_16), kind = 16)
  real(16), parameter :: rst9 = cmplx(huge(0.0_16), huge(0.0_16), kind = 16)
  real(16) :: rst10 = cmplx(-huge(0.0_16), -huge(0.0_16), kind = 16)
  real(16) :: rst11 = cmplx(-tiny(0.0_16), -tiny(0.0_16), kind = 16)
  real(16) :: rst12 = cmplx(-epsilon(0.0_16), -epsilon(0.0_16), kind = 16)
  real(16) :: rst13 = (-7.123456789012345678901234567890123456789_16,&
                                   -7.123456789012345678901234567890123456789_16)
  real(16) :: rst14 = (0.0_16, 0.0_16)
  real(16) :: rst15 = (77.123456789012345678901234567890123456789_16,&
                                   77.123456789012345678901234567890123456789_16)
  real(16) :: rst16 = cmplx(tiny(0.0_16), tiny(0.0_16), kind = 16)
  real(16) :: rst17 = cmplx(epsilon(0.0_16), epsilon(0.0_16), kind = 16)
  real(16) :: rst18 = cmplx(huge(0.0_16), huge(0.0_16), kind = 16)

  expect(1) = -1.18973149535723176508575932662800702E+4932_16 ! -real16_max
  expect(2) = -3.36210314311209350626267781732175260E-4932_16 ! -real16_min
  expect(3) = -1.92592994438723585305597794258492732E-0034_16 ! -real16_eps
  expect(4) = -7.123456789012345678901234567890123_16
  expect(5) = 0.0_16
  expect(6) = 77.12345678901234567890123456789012_16
  expect(7) = 3.36210314311209350626267781732175260E-4932_16 ! real16_min
  expect(8) = 1.92592994438723585305597794258492732E-0034_16 ! real16_eps
  expect(9) = 1.18973149535723176508575932662800702E+4932_16 ! real16_max
  expect(10) = -1.18973149535723176508575932662800702E+4932_16 ! -real16_max
  expect(11) = -3.36210314311209350626267781732175260E-4932_16 ! -real16_min
  expect(12) = -1.92592994438723585305597794258492732E-0034_16 ! -real16_eps
  expect(13) = -7.123456789012345678901234567890123_16
  expect(14) = 0.0_16
  expect(15) = 77.12345678901234567890123456789012_16
  expect(16) = 3.36210314311209350626267781732175260E-4932_16 ! real16_min
  expect(17) = 1.92592994438723585305597794258492732E-0034_16 ! real16_eps
  expect(18) = 1.18973149535723176508575932662800702E+4932_16 ! real16_max
  expect(19) = -1.18973149535723176508575932662800702E+4932_16 ! -real16_max
  expect(20) = -3.36210314311209350626267781732175260E-4932_16 ! -real16_min
  expect(21) = -1.92592994438723585305597794258492732E-0034_16 ! -real16_eps
  expect(22) = -7.123456789012345678901234567890123_16
  expect(23) = 0.0_16
  expect(24) = 77.12345678901234567890123456789012_16
  expect(25) = 3.36210314311209350626267781732175260E-4932_16 ! real16_min
  expect(26) = 1.92592994438723585305597794258492732E-0034_16 ! real16_eps
  expect(27) = 1.18973149535723176508575932662800702E+4932_16 ! real16_max
  expect(28) = -1.18973149535723176508575932662800702E+4932_16 ! -real16_max
  expect(29) = -3.36210314311209350626267781732175260E-4932_16 ! -real16_min
  expect(30) = -1.92592994438723585305597794258492732E-0034_16 ! -real16_eps
  expect(31) = -7.123456789012345678901234567890123_16
  expect(32) = 0.0_16
  expect(33) = 77.12345678901234567890123456789012_16
  expect(34) = 3.36210314311209350626267781732175260E-4932_16 ! real16_min
  expect(35) = 1.92592994438723585305597794258492732E-0034_16 ! real16_eps
  expect(36) = 1.18973149535723176508575932662800702E+4932_16 ! real16_max

  c_16(1) = cmplx(-huge(0.0_16), -huge(0.0_16), kind = 16)
  c_16(2) = cmplx(-tiny(0.0_16), -tiny(0.0_16), kind = 16)
  c_16(3) = cmplx(-epsilon(0.0_16), -epsilon(0.0_16), kind = 16)
  c_16(4) = (-7.123456789012345678901234567890123456789_16,&
             -7.123456789012345678901234567890123456789_16)
  c_16(5) = (0.0_16, 0.0_16)
  c_16(6) = (77.123456789012345678901234567890123456789_16,&
             77.123456789012345678901234567890123456789_16)
  c_16(7) = cmplx(tiny(0.0_16), tiny(0.0_16), kind = 16)
  c_16(8) = cmplx(epsilon(0.0_16), epsilon(0.0_16), kind = 16)
  c_16(9) = cmplx(huge(0.0_16), huge(0.0_16), kind = 16)

  do i = 1, n
    result(i) = c_16(i)
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
  result(n+10) = cmplx(-huge(0.0_16), -huge(0.0_16), kind = 16)
  result(n+11) = cmplx(-tiny(0.0_16), -tiny(0.0_16), kind = 16)
  result(n+12) = cmplx(-epsilon(0.0_16), -epsilon(0.0_16), kind = 16)
  result(n+13) = (-7.123456789012345678901234567890123456789_16,&
             -7.123456789012345678901234567890123456789_16)
  result(n+14) = (0.0_16, 0.0_16)
  result(n+15) = (77.123456789012345678901234567890123456789_16,&
             77.123456789012345678901234567890123456789_16)
  result(n+16) = cmplx(tiny(0.0_16), tiny(0.0_16), kind = 16)
  result(n+17) = cmplx(epsilon(0.0_16), epsilon(0.0_16), kind = 16)
  result(n+18) = cmplx(huge(0.0_16), huge(0.0_16), kind = 16)
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
    if (expect(i) .eq. 0.0_16) then
      if (result(i) .ne. expect(i)) STOP i
    else
      if (abs((result(i) - expect(i)) / expect(i)) .gt. q_tol) STOP i
    endif
  enddo

  print *, 'PASS'

end program
