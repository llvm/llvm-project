! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for negative operation of quad precision in flang2.

program test
  integer, parameter :: n = 3
  real(16) :: q1, q2, qa, qneg 
  real(16) :: result(n), expect(n) 
  qneg(qa) = -qa

  q1 = 1.0_16
  q2 = 2.0_16

  expect = (/ atan(1.0_16), 1.0_16, -2.0_16 /)

  ! op1 is constant
  result(1) = qneg(-atan(1.0_16))
  ! -(a - b)
  result(2) = -(q1 - q2)
  ! - a * b
  result(3) = - q1 * q2

  if (any(result /= expect)) STOP 1   
  print *, 'PASS'

end program
