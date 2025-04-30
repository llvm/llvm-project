! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
logical expect(12),rslt(12)
type :: stuff
integer :: kk1=selected_int_kind(2)
integer :: ll1 = 10
character(len=10) :: cc
integer dd
integer:: pp(10)
end type

type :: objects(k1,l1)
integer, kind:: k1=selected_int_kind(2)
integer, kind :: l1 = 10
character(len=l1) :: c
integer d
integer:: p(l1)
end type
contains
subroutine foo(x,n)
integer n
type(objects(l1=10)):: x

rslt(5) = x%c .eq. 'abcdefghij'
rslt(6) = x%l1 .eq. n 
rslt(7) = size(x%p) .eq. x%l1

rslt(8) = .true.
do i=1,size(x%p)
  if (x%p(i) .ne. i) rslt(8) = .false.
enddo

end subroutine

subroutine foo2(x,n)
integer n
type(objects):: x

rslt(9) = x%c .eq. 'abcdefghij'
rslt(10) = x%l1 .eq. n
rslt(11) = size(x%p) .eq. x%l1

rslt(12) = .true.
do i=1,size(x%p)
  if (x%p(i) .ne. i) rslt(12) = .false.
enddo

end subroutine


end module

program ppp
use mod
integer y 
type(objects(l1=10))::z
type(objects(l1=10)):: q

do i=1, size(z%p)
z%p(i) = i
enddo

z%c = 'abcdefghij'

rslt(1) = z%c .eq. 'abcdefghij'
rslt(2) = z%l1 .eq. 10
rslt(3) = size(z%p) .eq. z%l1

rslt(4) = .true.
do i=1,size(z%p)
  if (z%p(i) .ne. i) rslt(4) = .false.
enddo

q = z

call foo(q,10)
call foo2(z,10)

expect = .true.
call check(rslt,expect,12)

end
