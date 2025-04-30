!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program p
 type dt
  character(len=3) n
  character(len=3) m
  character(len=3) p
  real r
 end type
 type(dt),parameter:: alpha = dt('cad','cab','tar',2.0)
 character(len=3) n
 integer result(7), expect(7)
 data expect/99,101,99,101,100,99,101/

 n = 'car'
 select case(n)
 case(alpha%n:)
  result(1) = 99
 case(alpha%m)
  result(1) = 100
 case default
  result(1) = 101
 end select
  
 n = 'tee'
 select case(n)
 case(alpha%n:alpha%p)
  result(2) = 99
 case(alpha%m)
  result(2) = 100
 case default
  result(2) = 101
 end select
  
 n = 'tab'
 select case(n)
 case(alpha%n:alpha%p)
  result(3) = 99
 case(alpha%m)
  result(3) = 100
 case default
  result(3) = 101
 end select
  
 n = 'car'
 select case(n)
 case(alpha%n)
  result(4) = 99
 case(alpha%m)
  result(4) = 100
 case default
  result(4) = 101
 end select
  
 n = 'cab'
 select case(n)
 case(alpha%n)
  result(5) = 99
 case(alpha%m)
  result(5) = 100
 case default
  result(5) = 101
 end select
  
 n = 'pop'
 select case(n)
 case(alpha%n:'zebra')
  result(6) = 99
 case(alpha%m)
  result(6) = 100
 case default
  result(6) = 101
 end select
  
 n = 'pop'
 select case(n)
 case(alpha%n:'fez')
  result(7) = 99
 case(alpha%m)
  result(7) = 100
 case default
  result(7) = 101
 end select
  
 call check(result,expect,7)
end program
