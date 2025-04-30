!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!	check that passing intrinsics works
!	with or without interface blocks
!
	program tp0023
	intrinsic sin, cos
	real pi, res(4), exp(4)
	data exp/0.0,1.0,0.0,1.0/
	interface
	    subroutine  SUBR(RF1, ARG, RES )
	       real  ARG, RES
	       interface
		 function  RF1(RX)
		   real  RF1
		   real  RX
		 end function
	       end interface
	    end subroutine
	end interface

	pi = acos(1.0)
	call subr(sin, pi, res(1))
	call subr(cos, pi, res(2))
	call subs(sin, pi, res(3))
	call subs(cos, pi, res(4))
	call check(res,exp,4)
	end

	subroutine  SUBR( RF1, ARG, RES )
	  real  ARG, RES
	  interface
	    function  RF1(RX)
	      real  RF1
	      real  RX
	    end function
	  end interface
	  RES = RF1(ARG)
	end subroutine
	subroutine  SUBS( RF1, ARG, RES )
	  real  ARG, RES
	  external RF1
	  RES = RF1(ARG)
	end subroutine
