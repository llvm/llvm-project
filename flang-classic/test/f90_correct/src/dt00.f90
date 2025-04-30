!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!   Derived types

program p
type test
 integer,dimension(:,:),pointer:: mem
end type
integer results(9), expect(9)
data expect /4,5,6,7,99,9,10,99,12/

type(test)::a

allocate(a%mem(1:3,1:3))

do i = 1,3
 do j = 1,3
  a%mem(i,j) = i+j*3
 enddo
enddo

where(a%mem(1,:).gt.5) 
 a%mem(2,:) = 99
endwhere

do i = 1,3
 do j = 1,3
  k = i+(j-1)*3
  results(k) = a%mem(i,j)
 enddo
enddo
call check( results, expect, 9)
end
