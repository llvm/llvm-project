!*** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!*** See https://llvm.org/LICENSE.txt for license information.
!*** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! more derived type parameters in modules
	module md
	  type tscope
	     integer :: scope
	     integer :: junk
	  end type
	  type (tscope), parameter :: local = tscope(2,3)
	  type (tscope), parameter :: global = tscope(1,4)
	end module
	subroutine sub(a,x)
	  use md
	  type(tscope)::x
	  integer a
	  select case(x%scope)
	  case(local%scope)
	    a = 11
	  case(global%scope)
	    a = 22
	  case default
	    a = 33
	  end select
	  select case(x%junk)
	  case(local%junk)
	    a = a + 100
	  case(global%junk)
	    a = a + 200
	  case default
	    a = a + 300
	  end select
	end subroutine

	program p
	  use md
	  interface
	    subroutine sub(a,x)
	     use md
	     integer a
	     type(tscope)::x
	    end subroutine
	  end interface
	  type(tscope)::m
	  parameter(n=9)
	  integer result(n)
	  integer expect(n)
	  data expect/111,222,333,211,311,322,122,133,233/
	  m = tscope(2,3)
	  call sub(result(1),m)
	  m = tscope(1,4)
	  call sub(result(2),m)
	  m = tscope(3,5)
	  call sub(result(3),m)
	  m = tscope(2,4)
	  call sub(result(4),m)
	  m = tscope(2,5)
	  call sub(result(5),m)
	  m = tscope(1,2)
	  call sub(result(6),m)
	  m = tscope(1,3)
	  call sub(result(7),m)
	  m = tscope(9,3)
	  call sub(result(8),m)
	  m = tscope(9,4)
	  call sub(result(9),m)
	  call check(result, expect, n)
	end
