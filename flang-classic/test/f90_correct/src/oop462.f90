! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
type :: objects(l1,k1,l2)
integer , kind :: k1=4
integer(4), len :: l1 = 10
integer, len :: l2 = 20
integer :: z(l1)
end type
end module

program p
use mod
logical rslt(23)
logical expect(23)

type(objects(:)),allocatable :: x

allocate( objects(20) :: x )

expect = .true.

rslt(1) = allocated(x)
rslt(2) = x%l1 .eq. 20
rslt(3) = size(x%z,dim=1) .eq. 20 
do i=1, size(x%z)
  x%z(i) = i
enddo
do i=1, size(x%z)
  rslt(i+3) = x%z(i) .eq. i
enddo

call check(rslt,expect,23)


end
