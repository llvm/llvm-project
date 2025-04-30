! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for exponentiation operation of quad complex : complex(16) ^ real(8)

program test
  integer, parameter :: n = 6
  integer, parameter :: m = 2 * n
  integer :: i
  real(8) :: j = 777.77777777777777777_8, k = -17.7777777777777777_8, zero = 0.0_8,&
             one = 1.0_8, two = 2.0_8, half = 0.5_8
  complex(16) :: c1
  complex(16) :: rst(n)

  real(16) :: result(2*n), expect(2*n)

  c1 = (2.58879788522658767867845243213578_16, -1.3387123548975562114863159753156_16)

  rst(1) = c1 ** j
  rst(2) = c1 ** k
  rst(3) = c1 ** zero
  rst(4) = c1 ** one
  rst(5) = c1 ** two
  rst(6) = c1 ** half

  expect(1) = 1.856994246388429534129681604460917901E+0361_16
  expect(2) = -9.776556488125051164551164454207984935E+0360_16
  expect(3) = -3.248511934117149840587258072739469017E-0009_16
  expect(4) = 4.451498946170128539273854324782248952E-0009_16
  expect(5) = 1.0_16
  expect(6) = 0.0_16
  expect(7) = 2.58879788522658767867845243213578004_16
  expect(8) = -1.33871235489755621148631597531559997_16
  expect(9) = 4.90972372139829213745307352991854293_16
  expect(10) = -6.93131142657099727375757869256872649_16
  expect(11) = 1.65880212699080170099565297136762743_16
  expect(12) = -0.403517795496828286306937422438936510_16

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
