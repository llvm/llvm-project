! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for constant complex(16) convert to complex(8)

program test
  integer, parameter :: n = 9
  integer, parameter :: m = n * 4
  real(8), parameter :: d_tol = 5E-15
  complex(16) :: c_16(n)
  complex(8) :: result(m), expect(m)
  complex(8), parameter :: rst1 = cmplx(-huge(0.0_8), -huge(0.0_8), kind = 16)
  complex(8), parameter :: rst2 = cmplx(-tiny(0.0_8), -tiny(0.0_8), kind = 16)
  complex(8), parameter :: rst3 = cmplx(-epsilon(0.0_8), -epsilon(0.0_8), kind = 16)
  complex(8), parameter :: rst4 = (-7.123456789012345678901234567890123456789_16,&
                                   -7.123456789012345678901234567890123456789_16)
  complex(8), parameter :: rst5 = (0.0_16, 0.0_16)
  complex(8), parameter :: rst6 = (77.123456789012345678901234567890123456789_16,&
                                   77.123456789012345678901234567890123456789_16)
  complex(8), parameter :: rst7 = cmplx(tiny(0.0_8), tiny(0.0_8), kind = 16)
  complex(8), parameter :: rst8 = cmplx(epsilon(0.0_8), epsilon(0.0_8), kind = 16)
  complex(8), parameter :: rst9 = cmplx(huge(0.0_8), huge(0.0_8), kind = 16)
  complex(8) :: rst10 = cmplx(-huge(0.0_8), -huge(0.0_8), kind = 16)
  complex(8) :: rst11 = cmplx(-tiny(0.0_8), -tiny(0.0_8), kind = 16)
  complex(8) :: rst12 = cmplx(-epsilon(0.0_8), -epsilon(0.0_8), kind = 16)
  complex(8) :: rst13 = (-7.123456789012345678901234567890123456789_16,&
                                   -7.123456789012345678901234567890123456789_16)
  complex(8) :: rst14 = (0.0_16, 0.0_16)
  complex(8) :: rst15 = (77.123456789012345678901234567890123456789_16,&
                                   77.123456789012345678901234567890123456789_16)
  complex(8) :: rst16 = cmplx(tiny(0.0_8), tiny(0.0_8), kind = 16)
  complex(8) :: rst17 = cmplx(epsilon(0.0_8), epsilon(0.0_8), kind = 16)
  complex(8) :: rst18 = cmplx(huge(0.0_8), huge(0.0_8), kind = 16)

  expect(1) = (-1.7976931348623157E+308_8, -1.7976931348623157E+308_8) ! (-real8_max, -real8_max)
  expect(2) = (-2.2250738585072014E-308_8, -2.2250738585072014E-308_8) ! (-real8_min, -real8_min)
  expect(3) = (-2.2204460492503131E-016_8, -2.2204460492503131E-016_8) ! (-real8_eps, -real8_eps)
  expect(4) = (-7.123456789012345_8, -7.123456789012345_8)
  expect(5) = (0.0_8, 0.0_8)
  expect(6) = (77.12345678901235_8, 77.12345678901235_8)
  expect(7) = (2.2250738585072014E-308_8, 2.2250738585072014E-308_8) ! (real8_min, real8_min)
  expect(8) = (2.2204460492503131E-016_8, 2.2204460492503131E-016_8) ! (real8_eps, real8_eps)
  expect(9) = (1.7976931348623157E+308_8, 1.7976931348623157E+308_8) ! (real8_max, real8_max)
  expect(10) = (-1.7976931348623157E+308_8, -1.7976931348623157E+308_8) ! (-real8_max, -real8_max)
  expect(11) = (-2.2250738585072014E-308_8, -2.2250738585072014E-308_8) ! (-real8_min, -real8_min)
  expect(12) = (-2.2204460492503131E-016_8, -2.2204460492503131E-016_8) ! (-real8_eps, -real8_eps)
  expect(13) = (-7.123456789012345_8, -7.123456789012345_8)
  expect(14) = (0.0_8, 0.0_8)
  expect(15) = (77.12345678901235_8, 77.12345678901235_8)
  expect(16) = (2.2250738585072014E-308_8, 2.2250738585072014E-308_8) ! (real8_min, real8_min)
  expect(17) = (2.2204460492503131E-016_8, 2.2204460492503131E-016_8) ! (real8_eps, real8_eps)
  expect(18) = (1.7976931348623157E+308_8, 1.7976931348623157E+308_8) ! (real8_max, real8_max)
  expect(19) = (-1.7976931348623157E+308_8, -1.7976931348623157E+308_8) ! (-real8_max, -real8_max)
  expect(20) = (-2.2250738585072014E-308_8, -2.2250738585072014E-308_8) ! (-real8_min, -real8_min)
  expect(21) = (-2.2204460492503131E-016_8, -2.2204460492503131E-016_8) ! (-real8_eps, -real8_eps)
  expect(22) = (-7.123456789012345_8, -7.123456789012345_8)
  expect(23) = (0.0_8, 0.0_8)
  expect(24) = (77.12345678901235_8, 77.12345678901235_8)
  expect(25) = (2.2250738585072014E-308_8, 2.2250738585072014E-308_8) ! (real8_min, real8_min)
  expect(26) = (2.2204460492503131E-016_8, 2.2204460492503131E-016_8) ! (real8_eps, real8_eps)
  expect(27) = (1.7976931348623157E+308_8, 1.7976931348623157E+308_8) ! (real8_max, real8_max)
  expect(28) = (-1.7976931348623157E+308_8, -1.7976931348623157E+308_8) ! (-real8_max, -real8_max)
  expect(29) = (-2.2250738585072014E-308_8, -2.2250738585072014E-308_8) ! (-real8_min, -real8_min)
  expect(30) = (-2.2204460492503131E-016_8, -2.2204460492503131E-016_8) ! (-real8_eps, -real8_eps)
  expect(31) = (-7.123456789012345_8, -7.123456789012345_8)
  expect(32) = (0.0_8, 0.0_8)
  expect(33) = (77.12345678901235_8, 77.12345678901235_8)
  expect(34) = (2.2250738585072014E-308_8, 2.2250738585072014E-308_8) ! (real8_min, real8_min)
  expect(35) = (2.2204460492503131E-016_8, 2.2204460492503131E-016_8) ! (real8_eps, real8_eps)
  expect(36) = (1.7976931348623157E+308_8, 1.7976931348623157E+308_8) ! (real8_max, real8_max)

  c_16(1) = cmplx(-huge(0.0_8), -huge(0.0_8), kind = 16)
  c_16(2) = cmplx(-tiny(0.0_8), -tiny(0.0_8), kind = 16)
  c_16(3) = cmplx(-epsilon(0.0_8), -epsilon(0.0_8), kind = 16)
  c_16(4) = (-7.123456789012345678901234567890123456789_16,&
             -7.123456789012345678901234567890123456789_16)
  c_16(5) = (0.0_16, 0.0_16)
  c_16(6) = (77.123456789012345678901234567890123456789_16,&
             77.123456789012345678901234567890123456789_16)
  c_16(7) = cmplx(tiny(0.0_8), tiny(0.0_8), kind = 16)
  c_16(8) = cmplx(epsilon(0.0_8), epsilon(0.0_8), kind = 16)
  c_16(9) = cmplx(huge(0.0_8), huge(0.0_8), kind = 16)

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
  result(n+10) = cmplx(-huge(0.0_8), -huge(0.0_8), kind = 16)
  result(n+11) = cmplx(-tiny(0.0_8), -tiny(0.0_8), kind = 16)
  result(n+12) = cmplx(-epsilon(0.0_8), -epsilon(0.0_8), kind = 16)
  result(n+13) = (-7.123456789012345678901234567890123456789_16,&
             -7.123456789012345678901234567890123456789_16)
  result(n+14) = (0.0_16, 0.0_16)
  result(n+15) = (77.123456789012345678901234567890123456789_16,&
             77.123456789012345678901234567890123456789_16)
  result(n+16) = cmplx(tiny(0.0_8), tiny(0.0_8), kind = 16)
  result(n+17) = cmplx(epsilon(0.0_8), epsilon(0.0_8), kind = 16)
  result(n+18) = cmplx(huge(0.0_8), huge(0.0_8), kind = 16)
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
    if (expect(i)%re .eq. 0.0_8) then
      if (result(i)%re .ne. expect(i)%re) STOP i
    else
      if (abs((result(i)%re - expect(i)%re) / expect(i)%re) .gt. d_tol) STOP i
    endif
    if (expect(i)%im .eq. 0.0_8) then
      if (result(i)%im .ne. expect(i)%im) STOP i
    else
      if (abs((result(i)%im - expect(i)%im) / expect(i)%im) .gt. d_tol) STOP i
    endif
  enddo

  print *, 'PASS'

end program
