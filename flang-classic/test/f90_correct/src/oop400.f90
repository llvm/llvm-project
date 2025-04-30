! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
type :: objects(k1,k2)
integer, kind :: k1 = 3
integer, kind :: k2 = selected_char_kind("ASCII")
character(kind=k2,len=k1+k2) :: c
end type
end module

program p
use mod
logical expect(3), rslt(3)
type(objects) :: x

x%c = 'abcd'

expect = .true.

!print *, x%c, kind(x%c), len(x%c)
rslt(1) = kind(x%c) .eq. selected_char_kind("ASCII")
rslt(2) = len(x%c) .eq. 4 
rslt(3) = x%c .eq. 'abcd'

call check(rslt,expect,3)

end
