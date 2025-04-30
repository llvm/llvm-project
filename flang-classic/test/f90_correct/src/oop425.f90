! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
logical rslt(10),expect(10)
type :: objects(l1)
integer, len :: l1 = 5
integer :: p(l1)
character(len=l1) :: c
end type
contains
subroutine foo(x,n)
integer n
type(objects(10)):: x

rslt(5) = x%l1 .eq. 10

rslt(6) = .true.
do i=1, x%l1
if (x%p(i) .ne. i) rslt(6) = .false.
enddo

rslt(7) = size(x%p) .eq. x%l1

rslt(8) = x%c .eq. 'abcdefghij'

rslt(9) = len(x%c) .eq. x%l1

rslt(10) = len_trim(x%c) .eq. n

end subroutine

end module

program p
use mod
integer y 
type(objects(9+1))::z

expect = .true.
rslt = .false.

rslt(1) = z%l1 .eq. 10

do i=1, z%l1
z%p(i) = i
enddo

rslt(2) = .true.
do i=1, z%l1
if (z%p(i) .ne. i) rslt(2) = .false.
enddo

rslt(3) = size(z%p) .eq. z%l1

z%c = 'abcdefghij'

rslt(4) = z%c .eq. 'abcdefghij'

call foo(z,10)

call check(rslt,expect,10)

end
