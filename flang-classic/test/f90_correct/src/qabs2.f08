! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test ABS intrinsic with quad-precision arguments

program main
  integer(8), parameter :: a = abs(-123_8)
  real(4), parameter :: b = abs(-123.123_4)
  real(8), parameter :: c = abs(-456789456.456_8)
  real(16), parameter :: d = abs(-9999999.45678946546545_16)
  real(16), parameter :: e = abs(-9999999.45678946546545_16 + sin(0.0_16))
  real(16), parameter :: rst(2) = abs((/1.0_16, -2.0_16/))
  
  if(a /= 123_8) STOP 1
  if(b /= 123.123_4) STOP 2
  if(c /= 456789456.456_8) STOP 3
  if(d /= 9999999.45678946546545_16) STOP 4
  if(e /= 9999999.45678946546545_16) STOP 5
  if(rst(1) /= 1.0_16) STOP 6
  if(rst(2) /= 2.0_16) STOP 7

  write(*,*) 'PASS'
end
