!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

    module test

       integer N
       parameter (  N = 28)
       integer ND
       parameter (  ND= 4 )

      interface
	subroutine plainc (a,b,c) bind(c, name = 'newc')
	integer a,b
	real c
	end  subroutine plainc 
	
	subroutine intfunc (a,b,c,d,e,f,g,h,i) bind(c)
        integer, value :: a,b,c,d,e,f,g,h
        integer *8, value :: i
        end subroutine intfunc

	subroutine logfunc (a,b,c,d,e,f,g,h) bind(c)
        integer  , value :: a,b,c,d
        logical , value :: e,f,g,h
        end subroutine logfunc

	function realfunc (a,b,c,d,e,f,g,h,i) result(bind) bind(c)
        real, value :: a,b,c,e,f,g,h
        real *8, value :: i,d
	integer bind
        end function realfunc


	subroutine check (a,b,c) bind (c)
	integer a(N), b(N), c
        end subroutine check

      end interface



   common /result/ a_array
   integer a_array(N)
   BIND (c) ::/result/

   common /expect/ e_array
   integer e_array(N)
   BIND (c) ::/expect/


   common /d_expect/ d_array
   real *8 d_array(ND)
   BIND (c) ::/d_expect/

   common /d_result/ dr_array
   real *8 dr_array(ND)
   BIND (c) ::/d_result/

   commont

   end module

	use test
	logical ll
	real*8 tt
	real bind

      integer*8 kko
      kko = 45
	ll = .FALSE.
	tt = 50.0

       call plainc (1,2,4.0)
       call intfunc(3,4,5,6,10,20,30,40,kko)
       call logfunc(7,8,9,10,.TRUE., .FALSE., .TRUE. , .FALSE. )
       a_array(25) = realfunc(2.0, 3.0, 4.0, tt,3.0,3.0,6.0,30.03E04,400.004D01)
       call check(a_array, e_array , N)
       call checkd(dr_array, d_array, ND);
      end

