!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program testieee1
use ieee_exceptions
type(ieee_flag_type) :: ie1, ie2, ie3, ie4, ie5, ie6
type(ieee_flag_type) :: je1, je2, je3, je4, je5, je6
type(ieee_status_type) :: se1, sf1
logical l1(2), l2(2)
l1 = .true.
l2 = .true.
print *,"Test ieee derived types"
ie1 = ieee_underflow
ie2 = ieee_overflow
ie3 = ieee_divide_by_zero
ie4 = ieee_inexact
ie5 = ieee_invalid
ie6 = ieee_denorm
je1 = ie1
je2 = ie2
je3 = ie3
je4 = ie4
je5 = ie5
je6 = ie6
se1 = sf1
call ieee_get_status(se1)
sf1 = se1
call check(l1, l2, 2)
end
