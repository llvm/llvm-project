! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!   Internal functions

program p
 integer a,b
 parameter(n=1)
 integer result(n), expect(n)
 data expect/2/
 b = 1
 result(1) = f(b)
 call check( result, expect, n )
 
contains
 function f(x)
  integer f,x
  f = x + 1
 end function
end program
