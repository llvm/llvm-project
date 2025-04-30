! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for ieee_scalb, ieee_rint, ieee_logb with real*16

program ieee_test
  use ieee_arithmetic
  implicit none
  real(kind = 16) :: a, b, c, d
  logical :: resa(5), resb(14), resc(5), res(24), expct(24)
  integer(kind = 8) :: x
  real(kind = 16), parameter :: mx = 1.1897314953572317650857593266280070q+4932
  real(kind = 16), parameter :: mn = 3.3621031431120935062626778173217526q-4932
  real(kind = 16), parameter :: z0 = 2.5900700477752100443697755832911q-4965
  real(kind = 16), parameter :: z1 = 4.2435707662749041366954403156641q-4961
  real(kind = 16), parameter :: z2 = 2.7810665373859211750247237652736q-4956
  real(kind = 16), parameter :: z3 = 1.8225997659412373012642029668097q-4951
  real(kind = 16), parameter :: z4 = 1.1944589826072492777565080563284q-4946
  real(kind = 16), parameter :: z5 = 7.8280063884148688667050511979539q-4942
  real(kind = 16), parameter :: z6 = 5.1301622667115684604838223530911q-4937

  res = .false.
  expct = .true.

  a = 2.q0
  x = int(3, kind = 8)
  resa(1) = abs((ieee_scalb(a, x) - 16.0q0) / 16.0q0) .lt. 5.0q-33
  resa(2) = abs((ieee_scalb(a, int(2, kind = 8)) - 8.0q0) / 8.0q0) .lt. 5.0q-33
  resa(3) = .not. ieee_is_finite(ieee_scalb(3.0q0, int(32770, kind = 8)))
  resa(4) = ieee_scalb(3.0q0, int(-32770, kind = 8)) .lt. 5.0q-33
  resa(5) = abs((ieee_scalb(2.0q0, int(3, kind = 8)) - 16.0q0) / 16.0q0) .lt. 5.0q-33

  c = 0.0q0
  d = 1.0q0 / c

  resb(1) = .not. ieee_is_finite(ieee_logb(c))
  resb(2) = .not. ieee_is_finite(ieee_logb(d))
  resb(3) = ieee_logb(mx) .eq. 16383.0q0
  resb(4) = ieee_logb(mn) .eq. -16382.0q0
  resb(5) = ieee_logb(0.5q0) .eq. -1.0q0
  resb(6) = ieee_logb(8.0q0) .eq. 3.0q0
  resb(7) = ieee_logb(5.0q0) .eq. 2.0q0
  resb(8) = ieee_logb(z0) .eq. -16492.0q0
  resb(9) = ieee_logb(z1) .eq. -16478.0q0
  resb(10) = ieee_logb(z2) .eq. -16462.0q0
  resb(11) = ieee_logb(z3) .eq. -16446.0q0
  resb(12) = ieee_logb(z4) .eq. -16430.0q0
  resb(13) = ieee_logb(z5) .eq. -16414.0q0
  resb(14) = ieee_logb(z6) .eq. -16398.0q0

  b = 5.5q0

  resc(1) = ieee_rint(5.1q0) .eq. 5.0q0
  resc(2) = ieee_rint(b) .eq. 6.0q0
  resc(3) = ieee_rint(1.235q1) .eq. 12.0q0
  resc(4) = ieee_rint(-2.1q0) .eq. -2.0q0
  resc(5) = ieee_rint(-4.7q0) .eq. -5.0q0

  res(1:5) = resa
  res(6:19) = resb
  res(20:24) = resc

  call check(res, expct, 24)
end program
