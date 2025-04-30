! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Use option -Mq,10,2 to check ili of conversion from quad-precision floating-point
! to integer/integer64.

subroutine sub()
  integer(4) :: i4
  integer(8) :: i8
  real(16) :: r16
  i4 = r16
  i8 = r16
end subroutine

