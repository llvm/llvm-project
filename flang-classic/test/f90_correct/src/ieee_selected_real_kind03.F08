!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for ieee_selected_real_kind intrinsic.

program test
  use check_mod
  use ieee_arithmetic
#ifdef __flang_quadfp__
  integer, parameter :: N = 34
#else
  integer, parameter :: N = 26
#endif
  integer, dimension(N) :: rslt, expect
  integer(kind=1) :: argp
  integer(kind=2) :: argr
  integer(kind=8) :: argradix
  real(kind=4) :: r4
  real(kind=8) :: r8
#ifdef __flang_quadfp__
  real(kind=16) :: r16
#endif

  data expect / 4, 4, 4, 4, 4, 4, 4, 4, -1, 4, &
                8, 8, 8, 8, 8, 8, -1, 8, &
                -5, 4, -5, &
#ifdef __flang_quadfp__
                16, 16, -1, 16, 16, -2, &
                16, 16, -1, 16, -1, -2, -3/
#else
                -1, -2, -1, -2, -3/
#endif

  argp = 6_1
  argr = 37_2
  r4 = 3.14159e10_4
  rslt(1) = ieee_selected_real_kind(p=0)
  rslt(2) = ieee_selected_real_kind(p=2)
  rslt(3) = ieee_selected_real_kind(p=argp)
  rslt(4) = ieee_selected_real_kind(r=0)
  rslt(5) = ieee_selected_real_kind(r=20)
  rslt(6) = ieee_selected_real_kind(r=argr)
  rslt(7) = ieee_selected_real_kind(precision(0.0),range(0.0))
  rslt(8) = ieee_selected_real_kind(precision(0.0),radix(0.0))
  rslt(9) = ieee_selected_real_kind(range(0.0),radix(0.0))
  rslt(10) = ieee_selected_real_kind(precision(r4),range(r4),radix(r4))

  argp = 15_1
  argr = 307_2
  r8 = 3.1415926535986e100_8
  rslt(11) = ieee_selected_real_kind(p=10)
  rslt(12) = ieee_selected_real_kind(p=argp)
  rslt(13) = ieee_selected_real_kind(r=100)
  rslt(14) = ieee_selected_real_kind(r=argr)
  rslt(15) = ieee_selected_real_kind(precision(0.d0),range(0.d0))
  rslt(16) = ieee_selected_real_kind(precision(0.d0),radix(0.d0))
  rslt(17) = ieee_selected_real_kind(range(0.d0),radix(0.d0))
  rslt(18) = ieee_selected_real_kind(precision(r8),range(r8),radix(r8))

  argradix = 3_8
  rslt(19) = ieee_selected_real_kind(radix=1)
  rslt(20) = ieee_selected_real_kind(radix=2)
  rslt(21) = ieee_selected_real_kind(radix=argradix)

  argradix = 2_8
#ifdef __flang_quadfp__
  argp = 35_1
  argr = 4931_2
  r16 = 3.14159265358979323846264338327e300_16
  rslt(22) = ieee_selected_real_kind(p=25)
  rslt(23) = ieee_selected_real_kind(p=33)
  rslt(24) = ieee_selected_real_kind(p=argp)
  rslt(25) = ieee_selected_real_kind(r=1000)
  rslt(26) = ieee_selected_real_kind(r=argr)
  rslt(27) = ieee_selected_real_kind(r=5000)
  rslt(28) = ieee_selected_real_kind(precision(0.q0),range(0.q0))
  rslt(29) = ieee_selected_real_kind(precision(0.q0),radix(0.q0))
  rslt(30) = ieee_selected_real_kind(range(0.q0),radix(0.q0))
  rslt(31) = ieee_selected_real_kind(precision(r16),range(r16),radix(r16))
  rslt(32) = ieee_selected_real_kind(argp, 37, 2)
  rslt(33) = ieee_selected_real_kind(6, 5000, argradix)
  rslt(34) = ieee_selected_real_kind(argp, 5000, 2)
#else
  argp = 16_1
  argr = 308_2
  rslt(22) = ieee_selected_real_kind(p=argp)
  rslt(23) = ieee_selected_real_kind(r=argr)
  rslt(24) = ieee_selected_real_kind(argp, 37, 2)
  rslt(25) = ieee_selected_real_kind(6, argr, 2)
  rslt(26) = ieee_selected_real_kind(argp, argr, argradix)
#endif

  call checki4(rslt, expect, N)
end program
