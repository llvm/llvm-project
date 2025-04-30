!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!   forall containing derived type assignments


type junk
 integer m1,m2(2),m3
end type

integer result(20), expect(20)
data expect/10,20,30,40, 20,40,60,80, 30,60,90,120, &
40,80,120,160, 50,100,150,200 /

type(junk):: j(5)

forall(i=1:5)
 j(i)%m1 = 10*i
 j(i)%m2(1) = 20*i
 j(i)%m2(2) = 30*i
 j(i)%m3 = 40*i
endforall

do i = 1,5
 result(i*4-3) = j(i)%m1
 result(i*4-2) = j(i)%m2(1)
 result(i*4-1) = j(i)%m2(2)
 result(i*4-0) = j(i)%m3
enddo

!print *,result
call check( result, expect, 20 )

end
