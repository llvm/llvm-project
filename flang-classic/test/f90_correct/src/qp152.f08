! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for max and min take quad precision argument.

program test
  integer, parameter :: n = 7
  real(16) :: q2, q5, q15, qn5
  integer :: rslts(n), expect(n)
  
  data expect /-3, 3, 15, 5, 5, -2, -5/ 
  q2 = 2.0_16
  q5 = 5.0_16
  q15 = 1.5_16
  qn5 = -5.0_16

  rslts(1) = int(min(q2, q5)) + int(min(qn5, -1.0_16))
  rslts(2) = min(q15, q2+1.0_16) - max(-q15, qn5)
  rslts(3) = max(1.5_16, q2-1) * max(qn5*(-q2), 0.0_16)
  rslts(4) = max(q2, -q2, 2.5_16) * 2
  rslts(5) = q2 * min(-qn5, 2.6_16, -qn5)
  rslts(6) = min(5.0_16, 0.0_16, -q2)
  rslts(7) = min(q2, -q2, -2.5_16) * 2

  !check results:
  call check(rslts, expect, n)
   

end
