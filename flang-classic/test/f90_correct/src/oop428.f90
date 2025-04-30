! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
logical rslt(4),expect(4)
type :: stuff
integer :: kk1=selected_int_kind(2)
integer :: ll1 = 10
character(len=10) :: cc
integer dd
integer:: pp(10)
end type

type :: objects(l1,k1)
integer, kind:: k1=selected_int_kind(2)
integer, len :: l1 = 10
character(len=l1) :: c
integer :: p(l1)
integer d
end type

end module

program ppp
use mod
integer y 
type(objects)::z
!type(objects(l1=10)):: q

rslt(1) = z%l1 .eq. 10
do i=1,z%l1
  z%p(i)=i
enddo

rslt(2) = .true.
do i=1,z%l1
  if (z%p(i) .ne. i) rslt(2) = .false.
enddo


z%c = 'abcdefghij'
rslt(3) = z%c .eq. 'abcdefghij'

rslt(4) = size(z%p) .eq. z%l1

expect = .true.
call check(rslt,expect,4)

end
