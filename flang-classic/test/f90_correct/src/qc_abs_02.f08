!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!*  Test intrinsics function abs with quad precision complex in initialize.

program main
  use check_mod
  implicit none
  integer, parameter :: k = 16
  integer, parameter :: n = 10
  real*16, parameter :: maxr16 = sqrt(huge(1.0_16))
  real*16, parameter :: minr16 = tiny(1.0_16)
  !Maximum (For flang, vlaue of "real^2 + imag^2" must <= huge(1.0_16),
  !otherwise will make overflows).
  real(kind=k), parameter :: t1 = abs((0.0_16, maxr16))
  !Zero.
  real(kind=k), parameter :: t2 = abs((0.0_16, 0.0_16))
  real(kind=k), parameter :: t3 = abs((-0.0_16, -0.0_16))
  !Random number.
  real(kind=k), parameter :: t4 = abs((-9.123456770563732646435723823543532_16,&
                                       -1.012339992354898596734564736572374_16))
  real(kind=k), parameter :: t5 = abs((-0.0_16, maxr16))

  !Maximum.
  real(kind=k) :: t6 = abs((0.0_16, maxr16))
  !Zero.
  real(kind=k) :: t7 = abs((0.0_16, 0.0_16))
  real(kind=k) :: t8 = abs((-0.0_16, -0.0_16))
  !Random number.
  real(kind=k) :: t9 = abs((-9.123456770563732646435723823543532_16, &
                             -1.012339992354898596734564736572374_16))
  real(kind=k) :: t10 = abs((-0.0_16, maxr16))

  real(kind=k) :: rslt(n), expect(n)

  expect = (/ 1.09074813561941592946298424473378276E+2466_16, 0.0_16, 0.0_16,&
              9.17944964060843518655764186386383045_16,&
              1.09074813561941592946298424473378276E+2466_16,&
              1.09074813561941592946298424473378276E+2466_16, 0.0_16, 0.0_16,&
              9.17944964060843518655764186386383045_16,&
              1.09074813561941592946298424473378276E+2466_16 /)

  rslt(1) = t1
  rslt(2) = t2
  rslt(3) = t3
  rslt(4) = t4
  rslt(5) = t5
  rslt(6) = t6
  rslt(7) = t7
  rslt(8) = t8
  rslt(9) = t9
  rslt(10) = t10
  call checkr16(rslt, expect, n, rtoler = 1.0E-33_16)
end program main
