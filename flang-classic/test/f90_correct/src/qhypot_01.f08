! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test HYPOT intrinsic with quad-precision arguments

program test
  use ieee_arithmetic
 
  integer, parameter :: n = 5
  real(16), parameter :: q_tol = 1e-33_16
  real(16) :: x, y, answer
  integer :: rslts(n), expect(n)

  rslts = 0
  expect = 1

  !boundary
  answer = hypot(huge(0.0_16), huge(0.0_16))
  if (.not. ieee_is_finite(answer)) rslts(1) = 1
  answer = hypot(tiny(0.0_16), tiny(0.0_16))
  if (abs((answer - 4.75473186308633355902452847925549301E-4932_16) / answer) <= q_tol) rslts(2) = 1
  answer = hypot(epsilon(0.0_16), epsilon(0.0_16))
  if (abs((answer - 2.72367624753288964967419256279276029E-0034_16) / answer) <= q_tol) rslts(3) = 1

  !variable param
  x = 3.0_16
  y = 4.0_16
  answer = hypot(x, y)
  if (abs((answer - 5.0_16) / answer) <= q_tol) rslts(4) = 1
  
  !constant param
  answer = hypot(-3.7329_16, 1.4143_16)
  if (abs((answer - 3.99184003938033569095819983609610624_16) / answer) <= q_tol) rslts(5) = 1

  call check(rslts, expect, n)

end program test
