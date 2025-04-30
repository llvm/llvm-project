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
  integer :: select_rkind1 = ieee_selected_real_kind(p=0)
  integer :: select_rkind2 = ieee_selected_real_kind(p=2)
  integer :: select_rkind3 = ieee_selected_real_kind(p=6)
  integer :: select_rkind4 = ieee_selected_real_kind(r=0)
  integer :: select_rkind5 = ieee_selected_real_kind(r=20)
  integer :: select_rkind6 = ieee_selected_real_kind(r=37)
  integer :: select_rkind7 = ieee_selected_real_kind(precision(0.0),range(0.0))
  integer :: select_rkind8 = ieee_selected_real_kind(precision(0.0),radix(0.0))
  integer :: select_rkind9 = ieee_selected_real_kind(range(0.0),radix(0.0))
  integer :: select_rkind10 = &
      ieee_selected_real_kind(precision(0.0),range(0.0),radix(0.0))

  integer :: select_rkind11 = ieee_selected_real_kind(p=10)
  integer :: select_rkind12 = ieee_selected_real_kind(p=15)
  integer :: select_rkind13 = ieee_selected_real_kind(r=100)
  integer :: select_rkind14 = ieee_selected_real_kind(r=307)
  integer :: select_rkind15 = ieee_selected_real_kind(precision(0.d0),range(0.d0))
  integer :: select_rkind16 = ieee_selected_real_kind(precision(0.d0),radix(0.d0))
  integer :: select_rkind17 = ieee_selected_real_kind(range(0.d0),radix(0.d0))
  integer :: select_rkind18 = &
      ieee_selected_real_kind(precision(0.0d0),range(0.0d0),radix(0.0d0))

  integer :: select_rkind19 = ieee_selected_real_kind(radix=1)
  integer :: select_rkind20 = ieee_selected_real_kind(radix=2)
  integer :: select_rkind21 = ieee_selected_real_kind(radix=3)

#ifdef __flang_quadfp__
  integer :: select_rkind22 = ieee_selected_real_kind(p=25)
  integer :: select_rkind23 = ieee_selected_real_kind(p=33)
  integer :: select_rkind24 = ieee_selected_real_kind(p=35)
  integer :: select_rkind25 = ieee_selected_real_kind(r=1000)
  integer :: select_rkind26 = ieee_selected_real_kind(r=4931)
  integer :: select_rkind27 = ieee_selected_real_kind(r=5000)
  integer :: select_rkind28 = ieee_selected_real_kind(precision(0.q0),range(0.q0))
  integer :: select_rkind29 = ieee_selected_real_kind(precision(0.q0),radix(0.q0))
  integer :: select_rkind30 = ieee_selected_real_kind(range(0.q0),radix(0.q0))
  integer :: select_rkind31 = &
      ieee_selected_real_kind(precision(0.0q0),range(0.0q0),radix(0.0q0))
  integer :: select_rkind32 = ieee_selected_real_kind(35, 37, 2)
  integer :: select_rkind33 = ieee_selected_real_kind(6, 5000, 2)
  integer :: select_rkind34 = ieee_selected_real_kind(35, 5000, 2)
#else
  integer :: select_rkind22 = ieee_selected_real_kind(p=16)
  integer :: select_rkind23 = ieee_selected_real_kind(r=308)
  integer :: select_rkind24 = ieee_selected_real_kind(16, 37, 2)
  integer :: select_rkind25 = ieee_selected_real_kind(6, 308, 2)
  integer :: select_rkind26 = ieee_selected_real_kind(16, 308, 2)
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

  rslt(1) = select_rkind1
  rslt(2) = select_rkind2
  rslt(3) = select_rkind3
  rslt(4) = select_rkind4
  rslt(5) = select_rkind5
  rslt(6) = select_rkind6
  rslt(7) = select_rkind7
  rslt(8) = select_rkind8
  rslt(9) = select_rkind9
  rslt(10) = select_rkind10
  rslt(11) = select_rkind11
  rslt(12) = select_rkind12
  rslt(13) = select_rkind13
  rslt(14) = select_rkind14
  rslt(15) = select_rkind15
  rslt(16) = select_rkind16
  rslt(17) = select_rkind17
  rslt(18) = select_rkind18
  rslt(19) = select_rkind19
  rslt(20) = select_rkind20
  rslt(21) = select_rkind21
  rslt(22) = select_rkind22
  rslt(23) = select_rkind23
  rslt(24) = select_rkind24
  rslt(25) = select_rkind25
  rslt(26) = select_rkind26
#ifdef __flang_quadfp__
  rslt(27) = select_rkind27
  rslt(28) = select_rkind28
  rslt(29) = select_rkind29
  rslt(30) = select_rkind30
  rslt(31) = select_rkind31
  rslt(32) = select_rkind32
  rslt(33) = select_rkind33
  rslt(34) = select_rkind34
#endif

  call checki4(rslt, expect, N)
end program
