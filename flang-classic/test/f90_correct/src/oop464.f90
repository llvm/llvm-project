! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
logical expect(20), rslt(20)
type :: objects(k1,k2,l1)
integer, kind :: k1 = 3
integer, kind :: k2 = selected_char_kind("ASCII")
integer, len :: l1
character(kind=k2,len=l1) :: c
integer z
end type
contains
integer function check_dt(z,x)
type(objects(k2=1,k1=3,l1=:)),allocatable:: x
integer z
!type(objects(k2=1,k1=3,l1=*)):: x
!print *, x%c, kind(x%c), len(x%c), x%l1, x%z, z
rslt(14) = x%c .eq. 'abcd'
rslt(15) = kind(x%c) .eq. 1
rslt(16) = len(x%c) .eq. 4
rslt(17) = x%l1 .eq. 4
rslt(18) = x%z .eq. 999
rslt(19) = z .eq. 999
check_dt = x%z
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


x%z = 999
y%c = x%c
y%z = x%z
rslt(4) = x%c .eq. 'abcd'
rslt(5) = kind(x%c) .eq. 1
rslt(6) = len(x%c) .eq. 4
rslt(7) = x%l1 .eq. 4
rslt(8) = x%z .eq. 999
rslt(9) = y%c .eq. 'abcd'
rslt(10) = kind(y%c) .eq. 1
rslt(11) = len(y%c) .eq. 4
rslt(12) = y%l1 .eq. 4
rslt(13) = y%z .eq. 999

!print *, x%c, kind(x%c), len(x%c), x%l1, x%z
!print *, y%c, kind(y%c), len(y%c), y%l1, y%z
i = check_dt(x%z,x)
rslt(20) = i .eq. 999
!i = check_dt(y%z,y)
!print *, 'rslt =',i
call check(rslt,expect,20)

end
