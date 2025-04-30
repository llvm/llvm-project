!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program testieee1
use ieee_exceptions
logical l1(12), l2(12)
type(ieee_flag_type) :: ie1, ie2, ie3, ie4, ie5, ie6
print *,"Test ieee_support_halting"
ie1 = ieee_underflow
ie2 = ieee_overflow
ie3 = ieee_divide_by_zero
ie4 = ieee_inexact
ie5 = ieee_invalid
ie6 = ieee_denorm
l1 = .true.
l2 = .true.
l1(1 ) = ieee_support_halting(ieee_underflow)
l1(2 ) = ieee_support_halting(ieee_overflow)
l1(3 ) = ieee_support_halting(ieee_divide_by_zero)
l1(4 ) = ieee_support_halting(ieee_inexact)
l1(5 ) = ieee_support_halting(ieee_invalid)
l1(6 ) = ieee_support_halting(ieee_denorm)
l1(7 ) = ieee_support_halting(ie1)
l1(8 ) = ieee_support_halting(ie2)
l1(9 ) = ieee_support_halting(ie3)
l1(10) = ieee_support_halting(ie4)
l1(11) = ieee_support_halting(ie5)
l1(12) = ieee_support_halting(ie6)
call check(l1, l2, 12)
end
