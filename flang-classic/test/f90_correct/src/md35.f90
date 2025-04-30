!*
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!*
!     test interface sub with subroutine sub
	module interf
	  public  sub
	  interface  sub
	    module procedure  sub, subr
	  end interface
	
	  contains
	    subroutine  sub(II)
	      integer  II
	      II = 122
	    end subroutine  sub
	    subroutine  subr(rr)
	      real  rr
	      rr = 99
	    end subroutine  subr
	end module interf
	
	program  testmod
	  use  interf
	  implicit none
	  integer  II
	  integer,dimension(2) :: result,expect
	  data expect/122,99/
	  real  rr
	  ii = 0
	  rr = 0
	
	  call sub(II)
	  call sub(rr)
	  result(1) = ii
	  result(2) = rr
	  call check(result,expect,2)
	end
