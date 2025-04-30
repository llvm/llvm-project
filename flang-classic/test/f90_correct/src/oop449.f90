! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
logical expect(12),rslt(12)
type :: stuff(k11,k22)
integer,len :: k22 = 20
integer,kind :: k11 = 4
integer(k11) :: i = 3
integer(k11) :: j=4
integer :: p(k22)
end type
contains
subroutine foo(y,n)
integer n
type(stuff(2,n)) :: y

do i=1,y%k22
  y%p(i) = i
enddo


rslt(7) = y%i .eq. 3
rslt(8) = kind(y%i) .eq. 2
rslt(9) = y%k11 .eq. 2
rslt(10) = y%k22 .eq. size(y%p)
rslt(11) = y%j .eq. 4

rslt(12) = .true.
do i=1,y%k22
  if (y%p(i) .ne. i) rslt(12) = .false.
enddo
end subroutine


end module

program p
use mod
integer x

type(stuff(2,10)) :: y

do i=1,y%k22
 y%p(i) = i
enddo

rslt(1) = y%i .eq. 3
rslt(2) = kind(y%i) .eq. 2
rslt(3) = y%k11 .eq. 2 
rslt(4) = y%k22 .eq. 10
rslt(5) = y%j .eq. 4

rslt(6) = .true.
do i=1,y%k22
  if (y%p(i) .ne. i) rslt(6) = .false.
enddo

call foo(y,10)

expect = .true.
call check(rslt,expect,12)

end
