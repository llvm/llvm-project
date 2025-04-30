! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
type :: objects(k1,k2,l1)
integer, kind :: k2 = selected_char_kind("ASCII")
integer, kind :: k1 = selected_int_kind(4) + 6
integer, len :: l1 = 99
integer(kind=k1):: p(l1)
end type
end module

program p
use mod
type(objects(1,2,3)) :: x
logical expect(4), rslt(4)

rslt(1) = x%k1 .eq. 1
rslt(2) = x%k2 .eq. 2
rslt(3) = x%l1 .eq. 3
rslt(4) = size(x%p) .eq. x%l1


expect = .true.
call check(rslt,expect,3)


end
