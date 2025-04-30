! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for exponentiation operation of quad complex : complex(16) ^ real(16)

program test
  integer, parameter :: n = 16
  integer, parameter :: m = 2 * n
  integer :: i
  complex(16) :: c1, czero
  complex(16) :: rst1 = (0.58879788522658767867845243213578_16, -0.3387123548975562114863159753156_16) **&
                         77.7777777777777777777777777777777_16
  complex(16) :: rst2 = (2.58879788522658767867845243213578_16, -1.3387123548975562114863159753156_16) **&
                        -17.7777777777777777777777777777777_16
  complex(16) :: rst3 = (2.58879788522658767867845243213578_16, -1.3387123548975562114863159753156_16) ** 0.0_16
  complex(16) :: rst4 = (2.58879788522658767867845243213578_16, -1.3387123548975562114863159753156_16) ** 1.0_16
  complex(16) :: rst5 = (2.58879788522658767867845243213578_16, -1.3387123548975562114863159753156_16) ** 2.0_16
  complex(16) :: rst6 = (0.0_16, 0.0_16) ** 0.0_16
  complex(16) :: rst7 = (0.0_16, 0.0_16) ** 1.0_16
  complex(16) :: rst8 = (0.0_16, 0.0_16) ** 2.0_16

  complex(16), parameter :: rst9 = (-156.58879788522658767867845243213578_16, 338.41253312531532421342342141431_16) **&
                         77.7777777777777777777777777777777_16
  complex(16), parameter :: rst10 = (-223.58879788522658767878845243213578_16, 133.12321542153153214321432153124_16) **&
                        -17.7777777777777777777777777777777_16
  complex(16), parameter :: rst11 = (-1.58879788522658767867845243213578_16, 2.3387123548975562114863159753156_16) ** 0.0_16
  complex(16), parameter :: rst12 = (-1.58879788522658767867845243213578_16, 2.3387123548975562114863159753156_16) ** 1.0_16
  complex(16), parameter :: rst13 = (-1.58879788522658767867845243213578_16, 2.3387123548975562114863159753156_16) ** 2.0_16
  complex(16), parameter :: rst14 = (0.0_16, 0.0_16) ** 0.0_16
  complex(16), parameter :: rst15 = (0.0_16, 0.0_16) ** 1.0_16
  complex(16), parameter :: rst16 = (0.0_16, 0.0_16) ** 2.0_16
  real(16) :: result(2*n), expect(2*n)

  expect(1) = -8.398961555022714444534645025781261E-0014_16
  expect(2) = -2.040684179460875600148929979574491E-0014_16
  expect(3) = -3.248511934117150906722047821953665E-0009_16
  expect(4) = 4.451498946170133522570925452083600E-0009_16
  expect(5) = 1.0_16
  expect(6) = 0.0_16
  expect(7) = 2.58879788522658767867845243213578_16
  expect(8) = -1.33871235489755621148631597531560_16
  expect(9) = 4.90972372139829213745307352991854_16
  expect(10) = -6.93131142657099727375757869256872_16
  expect(11) = 1.0_16
  expect(12) = 0.0_16
  expect(13) = 0.0_16
  expect(14) = 0.0_16
  expect(15) = 0.0_16
  expect(16) = 0.0_16
  expect(17) = 3.725232015215296956263363298242071E+0199_16
  expect(18) = -9.565381544403295027835629332243588E+0199_16
  expect(19) = -7.840759294594200991903505047091153E-0044_16
  expect(20) = -8.410928726442897427680733301617846E-0044_16
  expect(21) = 1.0_16
  expect(22) = 0.0_16
  expect(23) = -1.58879788522658767867845243213578_16
  expect(24) = 2.33871235489755621148631597531560_16
  expect(25) = -2.94529675884999564287646328498422_16
  expect(26) = -7.43148248722906020814185160620909_16
  expect(27) = 1.0_16
  expect(28) = 0.0_16
  expect(29) = 0.0_16
  expect(30) = 0.0_16
  expect(31) = 0.0_16
  expect(32) = 0.0_16

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
    if(expect(i) .eq. 0.0_16) then
      if (result(i) .ne. expect(i)) STOP i
    else
      if (abs((result(i) - expect(i)) / expect(i)) .gt. 1E-33) STOP i
    endif
  enddo

  print *, 'PASS'
end
