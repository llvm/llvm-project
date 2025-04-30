! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test passing of quad-precision arguments by value

module m
  implicit none
contains
  subroutine test(q)
    real(16), value :: q
    q = -1.0_16
  end subroutine test
end module m

program p
  use m
  real(16) :: q1 = 1.0_16
  call test(q1)
  if (q1 /= 1.0_16) STOP 1
  write(*,*) 'PASS'
end
