! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
logical expect(9), rslt(9)
type :: objects(k1,k2,l1)
integer, kind :: k2 = selected_char_kind("ASCII")
integer, kind :: k1 = selected_int_kind(4)
integer,len :: l1 = 1000
integer(k1) :: p(l1)
end type
contains
subroutine foo(n)
integer n
type(objects) :: x

rslt(5) = x%k1 .eq. selected_int_kind(4) 
rslt(6) = x%k2 .eq. selected_char_kind("ASCII") 
rslt(7) = x%l1 .eq. n 
rslt(8) = size(x%p) .eq. x%l1


do i=1,n
x%p(i) = i
enddo

rslt(9) = .true.
do i=1,n
if (x%p(i) .ne. i) rslt(9) = .false.
enddo

end subroutine

end module

program p
use mod
integer y 
type(objects(1,20,30)) :: x

rslt(1) = x%k1 .eq. 1
rslt(2) = x%k2 .eq. 20
rslt(3) = x%l1 .eq. 30
rslt(4) = size(x%p) .eq. x%l1

call foo(1000)

expect = .true.
call check(rslt,expect,9)



end
