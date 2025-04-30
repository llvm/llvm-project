!*** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!*** See https://llvm.org/LICENSE.txt for license information.
!*** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! more derived type parameters in modules
	module md
	  type tscope
	     integer :: scope
	     real :: junk
	  end type
	  type (tscope), parameter :: local = tscope(2,3.333)
	  type (tscope), parameter :: global = tscope(1,4.444)
	end module
	subroutine sub(a,x)
	  use md
	  real,parameter::l = local%junk
	  real,parameter::g = global%junk
	  integer,parameter::mm = local%scope
	  real array(local%scope)
	  real a
	  integer x
	  select case(x)
	  case(local%scope)
	    a = l
	  case(global%scope)
	    a = g
	  case default
	    a = 0.0
	  end select
	end subroutine

	program p
	  use md
	  type(tscope)::m
	  parameter(n=3)
	  real result(n)
	  real expect(n)
	  data expect/3.333,4.444,0.000/
	  call sub(result(1),local%scope)
	  call sub(result(2),global%scope)
	  call sub(result(3),0)
	  call check(result, expect, n)
	end
