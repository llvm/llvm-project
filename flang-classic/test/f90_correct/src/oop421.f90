! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
logical expect(10), rslt(10)
type :: objects(l1)
integer(4), len :: l1 = 5
integer :: p(l1)
character(len=l1) :: c
end type
contains
subroutine foo(n)
integer n
type(objects(l1=n)),allocatable :: x

rslt(1) = .not.allocated(x)
allocate(x)
rslt(2) = x%l1 .eq. n
rslt(3) = allocated(x)
rslt(4) = size(x%p) .eq. n

do i=1,n
x%p(i) = i
enddo

rslt(5) = .true.
do i=1,n
if (x%p(i) .ne. i) rslt(5) = .false.
enddo

x%c = "1"
do i=1,n-1
x%c =  trim(x%c) // achar(iachar('1')+i)
enddo

rslt(6) = len_trim(x%c) .eq. n

rslt(7) = .true.
do i=1,n
if (x%c(i:i) .ne. achar(iachar('0')+i)) rslt(7) = .false.
enddo 

end subroutine

end module

program p
use mod
integer y 

rslt = .false.
call foo(25)

expect = .true.
call check(rslt,expect,7)



end
