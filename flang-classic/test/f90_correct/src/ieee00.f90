!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program testieee1
use ieee_exceptions
logical l1(32), l2(32)
real*4 a
real*8 d
type(ieee_flag_type) :: ie1, ie2, ie3, ie4, ie5, ie6
a = 1.0
d = 1.0d0
print *,"Test ieee_support_flag"
ie1 = ieee_underflow
ie2 = ieee_overflow
ie3 = ieee_divide_by_zero
ie4 = ieee_inexact
ie5 = ieee_invalid
ie6 = ieee_denorm
l1 = .true.
l2 = .true.
l1(2 ) = ieee_support_flag(ieee_underflow, a)
l1(3 ) = ieee_support_flag(ieee_underflow, d)
l1(4 ) = ieee_support_flag(ieee_overflow, a)
l1(5 ) = ieee_support_flag(ieee_overflow, d)
l1(6 ) = ieee_support_flag(ieee_divide_by_zero, a)
l1(7 ) = ieee_support_flag(ieee_divide_by_zero, d)
l1(8 ) = ieee_support_flag(ieee_inexact, a)
l1(9 ) = ieee_support_flag(ieee_inexact, d)
l1(10) = ieee_support_flag(ieee_invalid, a)
l1(11) = ieee_support_flag(ieee_invalid, d)
l1(12) = ieee_support_flag(ieee_denorm, a)
l1(13) = ieee_support_flag(ieee_denorm, d)
l1(14) = ieee_support_flag(ie1, a)
l1(15) = ieee_support_flag(ie1, d)
l1(16) = ieee_support_flag(ie2, a)
l1(17) = ieee_support_flag(ie2, d)
l1(18) = ieee_support_flag(ie3, a)
l1(19) = ieee_support_flag(ie3, d)
l1(20) = ieee_support_flag(ie4, a)
l1(21) = ieee_support_flag(ie4, d)
l1(22) = ieee_support_flag(ie5, a)
l1(23) = ieee_support_flag(ie5, d)
l1(24) = ieee_support_flag(ie6, a)
l1(25) = ieee_support_flag(ie6, d)
l1(26) = ieee_support_flag(ieee_underflow)
l1(27) = ieee_support_flag(ieee_overflow)
l1(28) = ieee_support_flag(ieee_divide_by_zero)
l1(29) = ieee_support_flag(ieee_inexact)
l1(30) = ieee_support_flag(ieee_invalid)
l1(31) = ieee_support_flag(ieee_denorm)

call check(l1, l2, 32)
end
