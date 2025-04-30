!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! check whether inlining with optional arguments works
! if the argument is not type scalar integer
! this test: inline a module subroutine into a module subroutine
module m
 real result(6)
 real expect(6)
 data expect/99,100,101,99,100,0/
 contains

 subroutine s1( a, b, c )
  real :: a
  real :: b
  real,optional :: c

  a = 99
  b = 100
  if( present(c) ) c = 101
 end subroutine

 subroutine s2
  real a
  data a/1/	! data statement prevents s2 from being inlined
		! we just want to inline s1
  result(6) = 0
  call s1( result(1), result(2), result(3) )
  call s1( result(4), result(5) )
 end subroutine
end module

program p
 use m
 call s2()
 !print *,result
 call check(result,expect,6)
end program
