!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

 subroutine s( a )
  integer a(10)
  do i = 1,10
   a(i) = a(i) + 1
  enddo
 end subroutine

module m
 integer,pointer :: x(:)
end module
 
subroutine t
 use m
 allocate(x(10))
 do i = 1,10
  x(i) = i+1
 enddo
 call s(x)
end subroutine

program p
 use m
 integer res(10),exp(10)
 data exp/3,4,5,6,7,8,9,10,11,12/
 call t()
 do i = 1,10
  res(i) = x(i)
 enddo
 call check(res,exp,10)
end program
