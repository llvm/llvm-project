! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
type :: objects(k1,k2,v1)
integer, kind :: k2 = selected_char_kind("ASCII")
integer, kind :: k1 = selected_int_kind(4) + 6
integer, len :: v1 = 99
end type
end module

program p
use mod
type(objects(1,2,3)) :: x
logical expect(3), rslt(3)

rslt(1) = x%k1 .eq. 1
rslt(2) = x%k2 .eq. 2
rslt(3) = x%v1 .eq. 3

expect = .true.
call check(rslt,expect,3)


end
