! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test SCALE intrinsic with quad-precision arguments

program main
  use check_mod
  use ieee_arithmetic
  real(4) :: a4(3), ea4(3)
  real(8) :: a8(3), ea8(3)
  real(16) :: a16(3), ea16(3), res(11), eres(11) 
  integer, parameter :: i = 30000
  real(16), parameter :: d = scale(3.0_16, 2)
  real(16), parameter :: rst1 = scale(1.0_16, -i)
  real(16), parameter :: rst2 = scale(1.0_16, i)
  real(16) :: e = scale(3.0_16,2000), dd, ee
  real(16) :: rst3 = scale(1.0_16, -i)
  real(16) :: rst4 = scale(1.0_16, i)

  a4 = [2.65, 0., tiny(1.)]
  a8 = [2.45646_8, 0.0_8, tiny(1.0_8)]
  a16 = [2.546798765414654654_16, 0.0_16, tiny(1.0_16)]
  ea4 = [3.3592742E+30, 0.000000, 1.4901161E-08]
  ea8 = [8.0409530926725562E+150_8, 0.000000000000000_8, &
         7.2835358703127019E-158_8]
  ea16 = [5.08102428299622831332366163584255075E+3010_16, &
         0.00000000000000000000000000000000000_16,       &
         6.70760797597180774202774403102198500E-1922_16]
  dd = 12.0000000000000000000000000000000000_16
  ee = 3.44439208582276357269849960353304595E+0602_16
  
  a4 = scale(a4, 100)
  a8 = scale(a8, 500)
  a16 = scale(a16, 10000)

  if (all(a4 .eq. ea4) .neqv. .true.) STOP 1
  if (all(a8 .eq. ea8) .neqv. .true.) STOP 1

  res(1:3) = a16
  res(4:4) = d
  res(5:5) = e
  res(6) = scale(1.0_16, -i) 
  res(7) = scale(1.0_16, i)
  res(8) = rst1
  res(9) = rst2
  res(10) = rst3
  res(11) = rst4
  eres(1:3) = ea16
  eres(4:4) = dd
  eres(5:5) = ee
  eres(6) = 0.0_16 
  eres(7) = ieee_value(eres(7), ieee_positive_inf) 
  eres(8) = 0.0_16 
  eres(9) = eres(7) 
  eres(10) = 0.0_16 
  eres(11) = eres(7) 
  call checkr16(res, eres, 11)
end program main
