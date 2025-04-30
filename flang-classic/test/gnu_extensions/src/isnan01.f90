!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
subroutine nan_check(is_nan, res)
  logical :: res(10)
  logical :: is_nan

  res(7) = is_nan
end subroutine nan_check

logical function nan_check2(is_nan)
  logical :: is_nan
  nan_check2 = is_nan
end

program isnan01
  use ieee_arithmetic
  integer, parameter :: N = 12
  integer :: exp(N), res(N)
  real ::nanf = 0.0, arr(1),temp(10)
  real(kind=8) :: nand =0.d0
  res = 1
  arr(1) = 1.0

  ! constant arguments
  if (isnan(0.0)) res(1) = 0
  if (isnan(0.d0)) res(2) = 0

  ! scalar float/ double values
  if (isnan(nanf)) res(3) = 0
  if (isnan(nand)) res(4) = 0
  if (isnan(arr(1))) res(5) = 0

  ! create NaNs
  nanf = nanf / 0.0
  nand = nand / 0.d0

  ! logical expressions.
  res(6) = isnan(nanf) .and. isnan(nand) .and. .true.

    ! as argument to subroutines.
  call nan_check(isnan(nanf), res)

  ! as function call
  res(8) = nan_check2(isnan(nand))

  ! more logical expressions
  if (.not. isnan(sqrt(-1.0))) res(9) = 0
  if (.not. isnan(sqrt(-1.d0))) res(10) = 0

  temp = 1.0
  if (any(isnan(temp))) res(11) = 0

  nanf = ieee_value(nanf,ieee_signaling_nan)
  if (.not. isnan(nanf)) res(12) = 0

  exp(1:N) = 1
  call check(res, exp, N)
end program
