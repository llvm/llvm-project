! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
logical expect(16), rslt(16)
type :: objects(k1,l1)
integer, kind :: k1 = selected_char_kind("ASCII")
integer, len :: l1
character(len=l1) :: c
character(len=l1) :: c2
integer z 
end type
contains
integer function foo(x)
!type(objects(l1=:)),allocatable :: x
type(objects(l1=*))::x
!print *, x%c, kind(x%c), len(x%c), x%l1, x%z, x%c2, len(x%c2)
rslt(10) = x%c .eq. 'abcd'
rslt(11) = kind(x%c) .eq. 1
rslt(12) = len(x%c) .eq. 4
rslt(13) = x%l1 .eq. 4
rslt(14) = x%z .eq. 88
rslt(15) = x%c2 .eq. 'defg'
rslt(16) = len(x%c2) .eq. 4
foo = x%l1
end function 
end module

program p
use mod
integer i
type(objects(l1=:)),allocatable :: x
type(objects(l1=4)) :: y

allocate(objects(l1=4) :: x)
x%c = 'abcd'
x%c2 = 'defg'
x%z = 88
rslt(1) = x%c .eq. 'abcd'
rslt(2) = kind(x%c) .eq. 1 
rslt(3) = len(x%c) .eq. 4
rslt(4) = x%l1 .eq. 4
rslt(5) = x%z .eq. 88
rslt(6) = x%c2 .eq. 'defg'
rslt(7) = len(x%c2) .eq. 4
!print *, x%c, kind(x%c), len(x%c), x%l1, x%z, x%c2, len(x%c2)
!print *, foo(x)
rslt(8) = foo(x) .eq. 4

y = x
!print *, foo(y)
rslt(9) = foo(y) .eq. 4
expect = .true.
call check(rslt,expect,16)

end
