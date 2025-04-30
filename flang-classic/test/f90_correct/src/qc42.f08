! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for function call of quad complex.

program test
  implicit none
  complex(16) :: r1, r2

  r1 = (0.0_16, 0.0_16)
  call sub(r2, r1 + (1.0_16, 2.0_16))
  if (r2 /= (1.1_16, 2.2_16)) STOP 1

  print *, 'PASS'
end

subroutine sub(o, i)
  complex(16) :: i, o
  o%re = i%re + 0.1_16
  o%im = i%im + 0.2_16
end

