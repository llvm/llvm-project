! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
implicit none
type :: objects(l1)
integer, len :: l1 = 10
character(len=l1) :: c
integer :: p(l1)
end type
end module

program p
use mod
implicit none
logical rslt(4),expect(4)
integer i
type(objects(10)) :: x

expect = .true.

x%c = '12345'
rslt(1) = trim(x%c) .eq. '12345'
rslt(2) = len(x%c) .eq. x%l1

do i=1,x%l1
  x%p(i) = i
enddo

rslt(3) = .true.
do i=1,x%l1
  if (x%p(i) .ne. i) rslt(3) = .false.
enddo

rslt(4) = size(x%p) .eq. x%l1

call check(rslt,expect,4)
end



