! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for form constant of quad complex.

program test
  integer, parameter :: n = 11
  integer :: i
  complex(16) :: result(n)

  result(1) = (1_1, 1.0_16)
  result(2) = (1_2, 1.0_16)
  result(3) = (1_4, 1.0_16)
  result(4) = (1_8, 1.0_16)
  result(5) = (1.0_4, 1.0_16)
  result(6) = (1.0_8, 1.0_16)
  result(7) = (1.0_16, 1.0_16)
  result(8) = (1.0_16, 1_1)
  result(9) = (1.0_16, 1_2)
  result(10) = (1.0_16, 1_4)
  result(11) = (1.0_16, 1_8)
  result(10) = (1.0_16, 1.0_4)
  result(11) = (1.0_16, 1.0_8)

  do i = 1, n
    if (result(i) /= (1.0_16, 1.0_16)) stop i
  enddo

  print *, 'PASS'
end
