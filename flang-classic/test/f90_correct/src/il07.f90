!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! check whether inlining with optional arguments works
! if the argument is not type scalar integer
! this test: inline a nonmodule subroutine into a nonmodule subroutine
!  the actual argument being a dummy
 subroutine s1( a, b, c )
  real :: a
  real :: b
  real,optional :: c

  a = 99
  b = 100
  if( present(c) ) c = 101
 end subroutine

 subroutine s2(result)
  real result(6)
  real a
  data a/1/	! data statement prevents s2 from being inlined
		! we just want to inline s1
  interface 
   subroutine s1( a, b, c )
    real :: a
    real :: b
    real,optional :: c
   end subroutine
  end interface
  result(6) = 0
  call s1( result(1), result(2), result(3) )
  call s1( result(4), result(5) )
 end subroutine

program p
 real result(6)
 real expect(6)
 data expect/99,100,101,99,100,0/
 call s2( result )
 !print *,result
 call check(result,expect,6)
end program
