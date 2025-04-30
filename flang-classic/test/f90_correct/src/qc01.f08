! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for exponentiation operation of double complex : complex(8) ^ complex(8)

program test
  integer, parameter :: n = 16
  integer, parameter :: m = 2 * n
  real(8), parameter :: d_tol = 5E-15_8
  integer :: i
  complex(8) :: c1, czero
  complex(8) :: rst1 = (0.5887897885226587_8, -0.3387123548975562_8) **&
                       (77.777777777777777_8, 6.1124388945234875_8)
  complex(8) :: rst2 = (2.5887978852265878_8, -1.3387863159753156_8) **&
                       (-17.77777777777777_8, -17.124057182309182_8)
  complex(8) :: rst3 = (2.5887845243213578_8, -1.3374863159753156_8) **&
                       (0.0_8, 0.0_8)
  complex(8) :: rst4 = (2.5887978853213578_8, -1.3387863159753156_8) **&
                       (1.0_8, 1.0_8)
  complex(8) :: rst5 = (2.5887978852213578_8, -1.3387163159753156_8) **&
                       (2.0_8, 2.0_8)
  complex(8) :: rst6 = (0.0_8, 0.0_8) ** (0.0_8, 0.0_8)
  complex(8) :: rst7 = (0.0_8, 0.0_8) ** (1.0_8, 0.0_8)
  complex(8) :: rst8 = (0.0_8, 0.0_8) ** (2.0_8, 0.0_8)

  complex(8), parameter :: rst9 = (-15.58879789852265_8, 33.49125331253153_8) **&
                                  (77.77777777777777_8, 12.15330942309482_8)
  complex(8), parameter :: rst10 = (-223.5885243213578_8, 133.12321432153124_8) **&
                                   (-17.77777777777777_8, -69.124311009919835_8)
  complex(8), parameter :: rst11 = (1.5887978243213578_8, 2.33871233159753156_8) **&
                                   (0.0_8, 0.0_8)
  complex(8), parameter :: rst12 = (1.5879784524313578_8, 2.33814863159753156_8) **&
                                   (1.0_8, 1.0_8)
  complex(8), parameter :: rst13 = (1.5887784543213578_8, 2.33874863159753156_8) **&
                                   (2.0_8, 2.0_8)
  complex(8), parameter :: rst14 = (0.0_8, 0.0_8) ** (0.0_8, 0.0_8)
  complex(8), parameter :: rst15 = (0.0_8, 0.0_8) ** (1.0_8, 0.0_8)
  complex(8), parameter :: rst16 = (0.0_8, 0.0_8) ** (2.0_8, 0.0_8)
  real(8) :: result(2*n), expect(2*n)

  expect(1) = 1.1068380071272079E-012_8
  expect(2) = 1.7841122924694556E-012_8
  expect(3) = -1.4273855376815681E-012_8
  expect(4) = 6.1683827705966633E-013_8
  expect(5) = 1.0_8
  expect(6) = 0.0_8
  expect(7) = 3.8967377649470918_8
  expect(8) = 2.6227800419402696_8
  expect(9) = 8.3046313110505618_8
  expect(10) = 20.439420030245451_8
  expect(11) = 1.0_8
  expect(12) = 0.0_8
  expect(13) = 0.0_8
  expect(14) = 0.0_8
  expect(15) = 0.0_8
  expect(16) = 0.0_8
  expect(17) = 8.8638462231772775E+110_8
  expect(18) = -1.9327383846031829E+111_8
  expect(19) = -1.6777129431219860E+035_8
  expect(20) = 5.9639109781757168E+034_8
  expect(21) = 1.0_8
  expect(22) = 0.0_8
  expect(23) = -0.45678246421495322_8
  expect(24) = 0.96423878016690179_8
  expect(25) = -0.72136710733385301_8
  expect(26) = -0.88200400379411836_8
  expect(27) = 1.0_8
  expect(28) = 0.0_8
  expect(29) = 0.0_8
  expect(30) = 0.0_8
  expect(31) = 0.0_8
  expect(32) = 0.0_8

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
    if(expect(i) .eq. 0.0_8) then
      if (result(i) .ne. expect(i)) STOP i
    else
      if (abs((result(i) - expect(i)) / expect(i)) .gt. d_tol) STOP i
    endif
  enddo

  print *, 'PASS'
end
