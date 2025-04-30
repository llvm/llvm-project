! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for exponentiation operation of quad complex : complex(16) ^ integer(4)

program test
  integer, parameter :: n = 9
  integer, parameter :: m = 2 * n
  integer :: i
  integer :: j = 777, k = -17, zero = 0, one = 1, two = 2
  complex(16) :: c1, czero
  complex(16) :: rst(n)

  real(16) :: result(2*n), expect(2*n)

  c1 = (2.58879788522658767867845243213578_16, -1.3387123548975562114863159753156_16)
  czero = (+0.0_16, -0.0_16)

  rst(1) = c1 ** j
  rst(2) = c1 ** k
  rst(3) = c1 ** zero
  rst(4) = c1 ** one
  rst(5) = c1 ** two
  rst(6) = czero ** j
  rst(7) = czero ** zero
  rst(8) = czero ** one
  rst(9) = czero ** two

  expect(1) = 9.074331812634794116775750627964576072E+0360_16
  expect(2) = -1.033534562999949754570831108142549999E+0360_16
  expect(3) = -3.24593241539185714088386393377543217E-0009_16
  expect(4) = 1.22398605966272405021565576156911277E-0008_16
  expect(5) = 1.0_16
  expect(6) = 0.0_16
  expect(7) = 2.58879788522658767867845243213578004_16
  expect(8) = -1.33871235489755621148631597531559997_16
  expect(9) = 4.90972372139829213745307352991854293_16
  expect(10) = -6.93131142657099727375757869256872649_16
  expect(11) = 0.0_16
  expect(12) = 0.0_16
  expect(13) = 1.0_16
  expect(14) = 0.0_16
  expect(15) = 0.0_16
  expect(16) = 0.0_16
  expect(17) = 0.0_16
  expect(18) = 0.0_16

  do i = 1, m, 2
    result(i) = rst((i+1)/2)%re
  enddo

  do i = 2, m, 2
    result(i) = rst(i/2)%im
  enddo

  do i = 1, m
    if(expect(i) .eq. 0.0_16) then
      if (result(i) .ne. expect(i)) STOP i
    else
      if (abs((result(i) - expect(i)) / expect(i)) .gt. 1E-33) STOP i
    endif
  enddo

  print *, 'PASS'
end
