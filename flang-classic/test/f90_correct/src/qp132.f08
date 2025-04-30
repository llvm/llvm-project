! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test the option -Hx,57,0x10 which maps REAL*16 to REAL*8, and prints a
! warning like "REAL(16) will be mapped to REAL(8)". This test will fail
! if the option "-Hx,57,0x10" is not specified.

program test
  integer, parameter :: n = 16
  real(kind = 16) :: r1 = 1.0_16
  real(16) :: r2 = 2.q0
  real*16, parameter :: r3 = 3.0q0
  real(16) :: r4 = huge(0.q0)
  real(16) :: r5 = tiny(0.q0)
  real(16) :: r6 = epsilon(0.q0)
  real(16) :: r7 = 1.234567890123456789012345678901234567890q0
  logical :: rst(n)

  rst(1) = precision(r1) == 15
  rst(2) = precision(r2) == 15
  rst(3) = precision(r3) == 15
  rst(4) = precision(4.0q123) == 15

  rst(5) = selected_real_kind(precision(r1)) == 8
  rst(6) = selected_real_kind(precision(r2)) == 8
  rst(7) = selected_real_kind(precision(r3)) == 8
  rst(8) = selected_real_kind(precision(4.0q123)) == 8

  rst(9) = r4 == huge(0.d0)
  rst(10) = r5 == tiny(0.d0)
  rst(11) = r6 == epsilon(0.d0)
  rst(12) = r7 == 1.2345678901234567d0

  rst(13) = (r1 + r7) == (1.0_8 + 1.2345678901234567d0)
  rst(14) = (r2 - r7) == (2.0_8 - 1.2345678901234567d0)
  rst(15) = (r3 * r7) == (3.0_8 * 1.2345678901234567d0)
  rst(16) = (4.q0 / r7) == (4.0_8 / 1.2345678901234567d0)

  if(any(rst .neqv. .true.)) STOP 1
  write(*,*) 'PASS'
end
