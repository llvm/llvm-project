! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test SQRT/EXP/LOG/SIN intrinsics with quad-precision arguments

program p
  use ISO_C_BINDING
  use check_mod

  interface
    subroutine get_expected_q( src, expct) bind(C)
    use ISO_C_BINDING
      type(C_PTR), value :: src
      type(C_PTR), value :: expct
    end subroutine
  end interface

  integer, parameter :: n = 15
  real (16) :: rslts(n) , expect(n)
  real (16) :: src(n), e1, e2, e3
  real (16) :: r1 = sqrt(1.23456_16)
  real (16), parameter :: r2 = sqrt(1.23456_16 + sin(0.0_16))
  real (16), parameter :: r3 = exp(1.23456_16 + sin(0.0_16))
  real (16), parameter :: r4 = log(1.23456_16 + sin(0.0_16))

  src(1) = 1111111111.22222222_16
  src(2) = 1.12345_16
  src(3) = 0.0_16
  src(4) = 145678.12345_16
  src(5) = 4.0_16
  src(6) = 11.22222222_16
  src(7) = -1.12345_16
  src(8) = 0.0_16
  src(9) = 2.12345_16
  src(10) = 3.0_16
  src(11) = 1111111111.22222222_16
  src(12) = 1.12345_16
  src(13) = 1.0_16
  src(14) = exp(3.0_16)
  src(15) = 1024.0_16

  rslts(1) = sqrt(src(1))
  rslts(2) = sqrt(src(2))
  rslts(3) = sqrt(src(3))
  rslts(4) = sqrt(145678.12345_16)
  rslts(5) = sqrt(4.0_16)

  rslts(6) = exp(src(6))
  rslts(7) = exp(src(7))
  rslts(8) = exp(src(8))
  rslts(9) = exp(2.12345_16)
  rslts(10) = exp(3.0_16)

  rslts(11) = log(src(11))
  rslts(12) = log(src(12))
  rslts(13) = log(src(13))
  rslts(14) = log(exp(3.0_16))
  rslts(15) = log(1024.0_16)

  e1 = 1.11110755554986664846214940411821916_16
  e2 = 3.43686596694224470265687660821204048_16
  e3 = 0.210714631295172515073249195797797214_16
  if(abs((r1 - e1) / r1) > 1e-33_16) stop 1
  if(abs((r2 - e1) / r2) > 1e-33_16) stop 2
  if(abs((r3 - e2) / r3) > 1e-33_16) stop 2
  if(abs((r4 - e3) / r4) > 1e-33_16) stop 2
  call get_expected_q(C_LOC(src), C_LOC(expect))
  call checkr16(rslts, expect, n)

end program
