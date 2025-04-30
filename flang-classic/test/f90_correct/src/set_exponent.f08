! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test SET_EXPONENT intrinsic with quad-precision arguments

program main
  use check_mod
  use ieee_arithmetic
  real(4) :: a4(3), ea4(3)
  real(4) :: inf4, nan4
  real(8) :: a8(3), ea8(3), inf8, nan8
  real(16) :: a16(5), ea16(5), inf16, nan16
  integer :: i = 30000
  inf4 = ieee_value(inf4, ieee_positive_inf)
  inf8 = ieee_value(inf8, ieee_positive_inf)
  inf16 = ieee_value(inf16, ieee_positive_inf)
  nan4 = ieee_value(nan4, ieee_quiet_nan)
  nan8 = ieee_value(nan8, ieee_quiet_nan)
  nan16 = ieee_value(nan16, ieee_quiet_nan)
  ea4 = [678.400024, 0.00000000, nan4]
  ea8 = [628.85375999999997_8, 0.0000000000000000_8, nan8]
  ea16 = [651.980483946151591424000000000000030_16,  &
          0.0_16, 0.0_16, inf16, nan16]

  a4 = [2.65, 0., inf4]
  a8 = [2.45646_8, 0.0_8, inf8]

  a4 = set_exponent(a4, 10)
  a8 = set_exponent(a8, 10)

  a16(1) = set_exponent(2.546798765414654654_16, 10)
  a16(2) = set_exponent(0.0_16, 10)
  a16(3) = set_exponent(1.0_16, -i)
  a16(4) = set_exponent(1.0_16, i)
  a16(5) = set_exponent(inf16, 10)

  if (ieee_is_finite(a4(3))) STOP 1
  if (ieee_is_finite(a8(3))) STOP 2
  if (ieee_is_finite(a16(5))) STOP 3

  if(all(a4(1:2) .eq. ea4(1:2)) .neqv. .true.) STOP 4
  if(all(a8(1:2) .eq. ea8(1:2)) .neqv. .true.) STOP 5
  if(all(a8(1:2) .eq. ea8(1:2)) .neqv. .true.) STOP 6

  !call checkr4(a4, ea4, 2)
  !call checkr8(a8, ea8, 2)
  call checkr16(a16, ea16, 4)

end program main
