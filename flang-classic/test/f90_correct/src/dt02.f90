!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program p
 type dt
  integer n
  integer m
  real r
 end type
 type(dt),parameter:: alpha = dt(1,3,2.0)
 integer n,x
 integer result(1), expect(1)
 data expect/99/

 n = 1
 select case(n)
 case(alpha%n)
  result(1) = 99
 case(alpha%m)
  result(1) = 100
 case default
  result(1) = 101
 end select
  
 call check(result,expect,1)
end program
