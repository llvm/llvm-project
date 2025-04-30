!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program p
 integer, dimension(3), parameter:: a = (/3,5,7/)
 integer n
 integer result(1), expect(1)
 data expect/100/

 n = 5
 select case(n)
 case(a(1))
  result(1) = 99
 case(a(2))
  result(1) = 100
 case default
  result(1) = 101
 end select
  
 call check(result,expect,1)
end program
