! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
logical expect(14), rslt(14)
type :: stuff(l1)
integer,kind:: l1 = 10
end type
type, extends(stuff) :: objects(l2)
integer,len:: l2 = 10
integer :: p(l1,l2)
end type
contains
subroutine foo(n)
integer n
type(objects(2,3)) :: x

rslt(1) = size(x%p,dim=1) .eq. x%l1
rslt(2) = size(x%p,dim=2) .eq. x%l2

k=0
do i=1,size(x%p,dim=1)
do j=1, size(x%p,dim=2)
  x%p(i,j) = k
  k = k + 1
enddo
enddo

k=0
rslt(3) = .true.
do i=1,size(x%p,dim=1)
do j=1, size(x%p,dim=2)
  if (x%p(i,j) .ne. k) then 
      print *, 'element ',i,',',j,' equals ',x%p(i,j),' but expecting ',k
      rslt(3) = .false.
  endif
  k = k + 1
enddo
enddo

end subroutine

end module

program p
use mod
integer y 

call foo(10)

expect = .true.
call check(rslt,expect,3)



end
