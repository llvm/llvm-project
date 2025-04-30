!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

 subroutine s( a )
  integer a(10,10)
  do i = 1,10
  do j = 1,10
   a(i,j) = a(i,j) + 1
  enddo
  enddo
 end subroutine

module m
 integer,pointer :: x(:,:)
end module
 
subroutine t
 use m
 allocate(x(10,10))
 do i = 1,10
 do j = 1,10
  x(i,j) = i+1 + (j-1)*100
 enddo
 enddo
 call s(x)
end subroutine

program p
 use m
 integer res(10,10),exp(10,10)
 data exp/ &
    3,  4,  5,  6,  7,  8,  9, 10, 11, 12, &
  103,104,105,106,107,108,109,110,111,112, &
  203,204,205,206,207,208,209,210,211,212, &
  303,304,305,306,307,308,309,310,311,312, &
  403,404,405,406,407,408,409,410,411,412, &
  503,504,505,506,507,508,509,510,511,512, &
  603,604,605,606,607,608,609,610,611,612, &
  703,704,705,706,707,708,709,710,711,712, &
  803,804,805,806,807,808,809,810,811,812, &
  903,904,905,906,907,908,909,910,911,912/
 call t()
 do i = 1,10
 do j = 1,10
  res(i,j) = x(i,j)
 enddo
 enddo
 call check(res,exp,100)
end program
