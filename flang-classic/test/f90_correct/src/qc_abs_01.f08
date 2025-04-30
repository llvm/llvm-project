!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!*  Test intrinsics function abs with quad precision complex.

program main
  use check_mod
  implicit none
  integer, parameter :: k = 16
  integer, parameter :: n = 12
  real*16, parameter :: maxr16 = huge(1.0_16)
  real*16, parameter :: minr16 = tiny(1.0_16)
  !Maximum.
  complex(kind=k), parameter :: t1 = (0, maxr16)
  !Zero.
  complex(kind=k), parameter :: t2 = (0.0_16, 0.0_16)
  complex(kind=k), parameter :: t3 = (-0.0_16, -0.0_16)
  !Random number.
  complex(kind=k), parameter :: t4 = (-9.123456770563732646435723823543532_16, &
                                      -1.012339992354898596734564736572374_16)
  complex(kind=k), parameter :: t5 = (-0.0_16, maxr16)
  !Minimum.
  complex(kind=k), parameter :: t6 = (minr16, minr16)

  complex(kind=k) :: t7, t8, t9, t10, t11, t12
  real(kind=k) :: rslt(n), expect(n)

  expect = (/ 1.18973149535723176508575932662800702E+4932_16, 0.0_16, 0.0_16,&
              9.17944964060843518655764186386383045_16, 1.18973149535723176508575932662800702E+4932_16,&
              4.75473186308633355902452847925549301E-4932_16,&
              1.18973149535723176508575932662800702E+4932_16, 0.0_16, 0.0_16,&
              9.17944964060843518655764186386383045_16, 1.18973149535723176508575932662800702E+4932_16,&
              4.75473186308633355902452847925549301E-4932_16 /)

  rslt(1) = abs(t1)
  rslt(2) = abs(t2)
  rslt(3) = abs(t3)
  rslt(4) = abs(t4)
  rslt(5) = abs(t5)
  rslt(6) = abs(t6)

  !Maximums.
  t7 = (0.0_16, maxr16)
  !Zero.
  t8 = (0.0_16, 0.0_16)
  t9 = (-0.0_16, -0.0_16)
  !Random number.
  t10 = (-9.123456770563732646435723823543532_16, &
        1.012339992354898596734564736572374_16)
  t11 = (-0.0, maxr16)
  !Minimums.
  t12 = (minr16, minr16)

  rslt(7) = abs(t7)
  rslt(8) = abs(t8)
  rslt(9) = abs(t9)
  rslt(10) = abs(t10)
  rslt(11) = abs(t11)
  rslt(12) = abs(t12)
  call checkr16(rslt, expect, n, rtoler = 1.0E-33_16)
end program main
