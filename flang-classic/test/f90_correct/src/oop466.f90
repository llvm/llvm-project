! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
logical expect(5), rslt(5)
type :: objects(k1,k2,l1)
integer, len :: l1
integer, kind :: k1 = 3
integer, kind :: k2 = selected_char_kind("ASCII")
character(kind=k2,len=l1) :: c
end type
contains
integer function check_dt(x)
type(objects(k2=1,k1=3,l1=:)),allocatable:: x
!print *, x%c
rslt(4) = x%c .eq. 'abcd'
check_dt = len(x%c)
end function

end module

program p
use mod
integer i
type(objects(l1=:)),allocatable :: x
type(objects(k2=1,k1=3,l1=4)) :: y

allocate(objects(k2=1,k1=3,l1=4) :: x)

x%c = 'abcd'

expect = .true.

!print *, x%c, kind(x%c), len(x%c)
rslt(1) = kind(x%c) .eq. selected_char_kind("ASCII")
rslt(2) = len(x%c) .eq. 4 
rslt(3) = x%c .eq. 'abcd'

y = x
!print *, y%c, kind(y%c), len(y%c), y%l1
i = check_dt(x)
!print *, 'len =',i
rslt(5) = i .eq. 4
call check(rslt,expect,5)

end
