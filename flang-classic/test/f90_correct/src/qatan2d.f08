! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test ATAN2D intrinsic with quad-precision arguments

program test
  
  real(16) :: y1, y2, x1, x2
  real(16) :: eps_q = 1.e-33_16
  integer :: result(2), expect(2)
  
  expect = 1
  result = 0

  x1 = 123.789789089_16
  x2 = 231.2327_16
  y1 = atan2d(123.789789089_16, 231.2327_16)
  y2 = atan2d(x1, x2)
  if (abs((y1 - 28.1622472844767945556550877128019815689521_16)/y1) <= eps_q) result(1) = 1
  if (abs((y2 - 28.1622472844767945556550877128019815689521_16)/y2) <= eps_q) result(2) = 1

  call check(result, expect, 2)

end program test
