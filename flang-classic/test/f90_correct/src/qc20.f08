! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for constant complex(4) convert to complex(16)

program test
  integer, parameter :: n = 9
  integer, parameter :: m = n * 4
  real(4), parameter :: f_tol = 5E-6
  complex(4) :: c_4(n)
  complex(16) :: result(m), expect(m)
  complex(16), parameter :: rst1 = cmplx(-huge(0.0_4), -huge(0.0_4), kind = 4)
  complex(16), parameter :: rst2 = cmplx(-tiny(0.0_4), -tiny(0.0_4), kind = 4)
  complex(16), parameter :: rst3 = cmplx(-epsilon(0.0_4), -epsilon(0.0_4), kind = 4)
  complex(16), parameter :: rst4 = (-7.123456_4, -7.123456_4)
  complex(16), parameter :: rst5 = (0.0_4, 0.0_4)
  complex(16), parameter :: rst6 = (77.12345_4, 77.12345_4)
  complex(16), parameter :: rst7 = cmplx(tiny(0.0_4), tiny(0.0_4), kind = 4)
  complex(16), parameter :: rst8 = cmplx(epsilon(0.0_4), epsilon(0.0_4), kind = 4)
  complex(16), parameter :: rst9 = cmplx(huge(0.0_4), huge(0.0_4), kind = 4)
  complex(16) :: rst10 = cmplx(-huge(0.0_4), -huge(0.0_4), kind = 4)
  complex(16) :: rst11 = cmplx(-tiny(0.0_4), -tiny(0.0_4), kind = 4)
  complex(16) :: rst12 = cmplx(-epsilon(0.0_4), -epsilon(0.0_4), kind = 4)
  complex(16) :: rst13 = (-7.123456_4, -7.123456_4)
  complex(16) :: rst14 = (0.0_4, 0.0_4)
  complex(16) :: rst15 = (77.12345_4, 77.12345_4)
  complex(16) :: rst16 = cmplx(tiny(0.0_4), tiny(0.0_4), kind = 4)
  complex(16) :: rst17 = cmplx(epsilon(0.0_4), epsilon(0.0_4), kind = 4)
  complex(16) :: rst18 = cmplx(huge(0.0_4), huge(0.0_4), kind = 4)

  expect(1) = (-3.402823466385288598117041834845169254E+0038_16, -3.402823466385288598117041834845169254E+0038_16) ! (-real4_max, -real4_max)
  expect(2) = (-1.175494350822287507968736537222245678E-0038_16, -1.175494350822287507968736537222245678E-0038_16) ! (-real4_min, -real4_min)
  expect(3) = (-1.192092895507812500000000000000000000E-0007_16, -1.192092895507812500000000000000000000E-0007_16) ! (-real4_eps, -real4_eps)
  expect(4) = (-7.12345600128173828125000000000000000_16, -7.12345600128173828125000000000000000_16)
  expect(5) = (0.0_16, 0.0_16)
  expect(6) = (77.1234512329101562500000000000000000_16, 77.1234512329101562500000000000000000_16)
  expect(7) = (1.175494350822287507968736537222245678E-0038_16, 1.175494350822287507968736537222245678E-0038_16) ! (real4_min, real4_min)
  expect(8) = (1.192092895507812500000000000000000000E-0007_16, 1.192092895507812500000000000000000000E-0007_16) ! (real4_eps, real4_eps)
  expect(9) = (3.402823466385288598117041834845169254E+0038_16, 3.402823466385288598117041834845169254E+0038_16) ! (real4_max, real4_max)
  expect(10) = (-3.402823466385288598117041834845169254E+0038_16, -3.402823466385288598117041834845169254E+0038_16) ! (-real4_max, -real4_max)
  expect(11) = (-1.175494350822287507968736537222245678E-0038_16, -1.175494350822287507968736537222245678E-0038_16) ! (-real4_min, -real4_min)
  expect(12) = (-1.192092895507812500000000000000000000E-0007_16, -1.192092895507812500000000000000000000E-0007_16) ! (-real4_eps, -real4_eps)
  expect(13) = (-7.12345600128173828125000000000000000_16, -7.12345600128173828125000000000000000_16)
  expect(14) = (0.0_16, 0.0_16)
  expect(15) = (77.1234512329101562500000000000000000_16, 77.1234512329101562500000000000000000_16)
  expect(16) = (1.175494350822287507968736537222245678E-0038_16, 1.175494350822287507968736537222245678E-0038_16) ! (real4_min, real4_min)
  expect(17) = (1.192092895507812500000000000000000000E-0007_16, 1.192092895507812500000000000000000000E-0007_16) ! (real4_eps, real4_eps)
  expect(18) = (3.402823466385288598117041834845169254E+0038_16, 3.402823466385288598117041834845169254E+0038_16) ! (real4_max, real4_max)
  expect(19) = (-3.402823466385288598117041834845169254E+0038_16, -3.402823466385288598117041834845169254E+0038_16) ! (-real4_max, -real4_max)
  expect(20) = (-1.175494350822287507968736537222245678E-0038_16, -1.175494350822287507968736537222245678E-0038_16) ! (-real4_min, -real4_min)
  expect(21) = (-1.192092895507812500000000000000000000E-0007_16, -1.192092895507812500000000000000000000E-0007_16) ! (-real4_eps, -real4_eps)
  expect(22) = (-7.12345600128173828125000000000000000_16, -7.12345600128173828125000000000000000_16)
  expect(23) = (0.0_16, 0.0_16)
  expect(24) = (77.1234512329101562500000000000000000_16, 77.1234512329101562500000000000000000_16)
  expect(25) = (1.175494350822287507968736537222245678E-0038_16, 1.175494350822287507968736537222245678E-0038_16) ! (real4_min, real4_min)
  expect(26) = (1.192092895507812500000000000000000000E-0007_16, 1.192092895507812500000000000000000000E-0007_16) ! (real4_eps, real4_eps)
  expect(27) = (3.402823466385288598117041834845169254E+0038_16, 3.402823466385288598117041834845169254E+0038_16) ! (real4_max, real4_max)
  expect(28) = (-3.402823466385288598117041834845169254E+0038_16, -3.402823466385288598117041834845169254E+0038_16) ! (-real4_max, -real4_max)
  expect(29) = (-1.175494350822287507968736537222245678E-0038_16, -1.175494350822287507968736537222245678E-0038_16) ! (-real4_min, -real4_min)
  expect(30) = (-1.192092895507812500000000000000000000E-0007_16, -1.192092895507812500000000000000000000E-0007_16) ! (-real4_eps, -real4_eps)
  expect(31) = (-7.12345600128173828125000000000000000_16, -7.12345600128173828125000000000000000_16)
  expect(32) = (0.0_16, 0.0_16)
  expect(33) = (77.1234512329101562500000000000000000_16, 77.1234512329101562500000000000000000_16)
  expect(34) = (1.175494350822287507968736537222245678E-0038_16, 1.175494350822287507968736537222245678E-0038_16) ! (real4_min, real4_min)
  expect(35) = (1.192092895507812500000000000000000000E-0007_16, 1.192092895507812500000000000000000000E-0007_16) ! (real4_eps, real4_eps)
  expect(36) = (3.402823466385288598117041834845169254E+0038_16, 3.402823466385288598117041834845169254E+0038_16) ! (real4_max, real4_max)

  c_4(1) = cmplx(-huge(0.0_4), -huge(0.0_4), kind = 4)
  c_4(2) = cmplx(-tiny(0.0_4), -tiny(0.0_4), kind = 4)
  c_4(3) = cmplx(-epsilon(0.0_4), -epsilon(0.0_4), kind = 4)
  c_4(4) = (-7.123456_4, -7.123456_4)
  c_4(5) = (0.0_4, 0.0_4)
  c_4(6) = (77.12345_4, 77.12345_4)
  c_4(7) = cmplx(tiny(0.0_4), tiny(0.0_4), kind = 4)
  c_4(8) = cmplx(epsilon(0.0_4), epsilon(0.0_4), kind = 4)
  c_4(9) = cmplx(huge(0.0_4), huge(0.0_4), kind = 4)

  do i = 1, n
    result(i) = c_4(i)
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
  result(n+10) = cmplx(-huge(0.0_4), -huge(0.0_4), kind = 4)
  result(n+11) = cmplx(-tiny(0.0_4), -tiny(0.0_4), kind = 4)
  result(n+12) = cmplx(-epsilon(0.0_4), -epsilon(0.0_4), kind = 4)
  result(n+13) = (-7.123456_4, -7.123456_4)
  result(n+14) = (0.0_4, 0.0_4)
  result(n+15) = (77.12345_4, 77.12345_4)
  result(n+16) = cmplx(tiny(0.0_4), tiny(0.0_4), kind = 4)
  result(n+17) = cmplx(epsilon(0.0_4), epsilon(0.0_4), kind = 4)
  result(n+18) = cmplx(huge(0.0_4), huge(0.0_4), kind = 4)
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
    if (expect(i)%re .eq. 0.0_16) then
      if (result(i)%re .ne. expect(i)%re) STOP i
    else
      if (abs((result(i)%re - expect(i)%re) / expect(i)%re) .gt. f_tol) STOP i
    endif
    if (expect(i)%im .eq. 0.0_16) then
      if (result(i)%im .ne. expect(i)%im) STOP i
    else
      if (abs((result(i)%im - expect(i)%im) / expect(i)%im) .gt. f_tol) STOP i
    endif
  enddo

  print *, 'PASS'

end program
