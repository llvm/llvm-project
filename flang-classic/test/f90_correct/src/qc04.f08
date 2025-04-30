! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for exponentiation operation of quad complex : complex(16) ^ integer(4)

program test
  integer, parameter :: n = 9
  integer, parameter :: m = 2 * n
  integer :: i
  complex(16) :: c1, czero
  complex(16) :: rst1 = (0.58879788522658767867845243213578_16, -0.3387123548975562114863159753156_16) ** 77
  complex(16) :: rst2 = (2.58879788522658767867845243213578_16, -1.3387123548975562114863159753156_16) ** -17
  complex(16) :: rst3 = (2.58879788522658767867845243213578_16, -1.3387123548975562114863159753156_16) ** 0
  complex(16) :: rst4 = (2.58879788522658767867845243213578_16, -1.3387123548975562114863159753156_16) ** 1
  complex(16) :: rst5 = (2.58879788522658767867845243213578_16, -1.3387123548975562114863159753156_16) ** 2
  complex(16) :: rst6 = (+0.0_16, -0.0_16) ** 777
  complex(16) :: rst7 = (+0.0_16, -0.0_16) ** 0
  complex(16) :: rst8 = (+0.0_16, -0.0_16) ** 1
  complex(16) :: rst9 = (+0.0_16, -0.0_16) ** 2
  real(16) :: result(2*n), expect(2*n)

  expect(1) = -9.33510564214525153948027376625749087E-0014_16
  expect(2) = -7.01406348823044624303544879940859472E-0014_16
  expect(3) = -3.24593241539185714088386393377543433E-0009_16
  expect(4) = 1.22398605966272405021565576156911263E-0008_16
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

  do i = 1, m
    if(expect(i) .eq. 0.0_16) then
      if (result(i) .ne. expect(i)) STOP i
    else
      if (abs((result(i) - expect(i)) / expect(i)) .gt. 1E-33) STOP i
    endif
  enddo

  print *, 'PASS'
end
