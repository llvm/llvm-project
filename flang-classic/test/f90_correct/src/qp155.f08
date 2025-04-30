! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for subtraction operation of quad precision in flang2.

program test
  integer, parameter :: n = 4
  real(16) :: q1, q2, qa1, qa2, qsub
  real(16) :: result(n), expect(n)
  qsub(qa1, qa2) = qa1 - qa2

  q1 = 1.0_16
  q2 = 2.0_16

  expect = (/ 1.0_16, -2.0_16, -1.0_16, 3.0_16 /)

  ! op2 is 0.0
  result(1) = q1 - 0.0_16
  ! op1 is 0.0
  result(2) = 0.0_16 - q2
  ! op1 and op2 is constant
  result(3) = qsub(1.0_16, 2.0_16)
  ! a - -b
  result(4) = q1 - (-q2)

  if (any(result /= expect)) STOP 1

  print *, 'PASS'

end program
