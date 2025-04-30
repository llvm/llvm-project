! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test various intrinsics with quad-precision arguments

program test
  integer, parameter :: n = 10
  integer, parameter :: m = n * 2
  real(16), parameter :: q_tol = 5E-33_16
  real(16) :: r(n/2) 
  real(16) :: result(m), expect(m) 

  r(1) = huge(0.0_16)
  r(2) = tiny(0.0_16)
  r(3) = epsilon(0.0_16)
  r(4) = 7.123456_16
  r(5) = 0.0_16

  result(1) = erf(r(1))
  result(2) = erf(r(2))
  result(3) = erf(r(3))
  result(4) = erf(r(4))
  result(5) = erf(r(5))
  result(6) = erfc(r(1))
  result(7) = erfc(r(2))
  result(8) = erfc(r(3))
  result(9) = erfc(r(4))
  result(10) = erfc(r(5))
  result(11) = erf(-huge(0.0_16))
  result(12) = erf(-tiny(0.0_16))
  result(13) = erf(-epsilon(0.0_16))
  result(14) = erf(-7.123456_16)
  result(15) = erf(-0.0_16)
  result(16) = erfc(-huge(0.0_16))
  result(17) = erfc(-tiny(0.0_16))
  result(18) = erfc(-epsilon(0.0_16))
  result(19) = erfc(-7.123456_16)
  result(20) = erfc(-0.0_16)

  expect(1) = 1.0_16
  expect(2) = 3.79372714431402898312579162181147078E-4932_16
  expect(3) = 2.17317922653197604343594986635858443E-0034_16
  expect(4) = 0.999999999999999999999992807512053531_16
  expect(5) = 0.0_16
  expect(6) = 0.0_16
  expect(7) = 1.0_16
  expect(8) = 0.999999999999999999999999999999999807_16
  expect(9) = 7.19248794642962870598754100808714111E-0024_16
  expect(10) = 1.0_16
  expect(11) = -1.0_16
  expect(12) = -3.79372714431402898312579162181147078E-4932_16
  expect(13) = -2.17317922653197604343594986635858443E-0034_16
  expect(14) = -0.999999999999999999999992807512053531_16
  expect(15) = -0.0_16
  expect(16) = 2.0_16
  expect(17) = 1.0_16
  expect(18) = 1.0_16 
  expect(19) = 1.99999999999999999999999280751205363_16
  expect(20) = 1.0_16
  do i = 1, m
    if (expect(i) .eq. 0.0_16) then
      if (result(i) .ne. expect(i)) STOP i
    else
      if (abs((result(i) - expect(i)) / expect(i)) .gt. q_tol) STOP i
    endif
  enddo 
 
  print *, 'PASS'

end program
