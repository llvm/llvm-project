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
 type(dt),parameter:: alpha = dt(8,9,2.0)
 integer n,x
 integer result(2), expect(2)
 data expect/100,101/

 n = 5
 select case(n)
 case(alpha%m)
  result(1) = 99
 case(:alpha%n)
  result(1) = 100
 case default
  result(1) = 101
 end select
  
 n = -2
 select case(n)
 case(alpha%m)
  result(2) = 99
 case(-1:alpha%n)
  result(2) = 100
 case default
  result(2) = 101
 end select
  
 call check(result,expect,2)
end program
