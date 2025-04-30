!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!   check that order of integer and pointer statements don't matter

	subroutine  sub1(x,n,m)
	  integer x(:),n,m
	  pointer x
	  allocate(x(1:n))
	  x = m
	end subroutine
	subroutine  sub2(x,n,m)
	  pointer x
	  integer x(:),n,m
	  allocate(x(1:n))
	  x = m
	end subroutine
	subroutine  sub3(x,n,m)
	  integer,pointer:: x(:)
	  integer n,m
	  allocate(x(1:n))
	  x = m
	end subroutine
	subroutine  sub4(x,n,m)
	  integer,dimension(:),pointer:: x
	  allocate(x(1:n))
	  x = m
	end subroutine
	subroutine  sub5(x,n,m)
	  integer,pointer,dimension(:):: x
	  integer n,m
	  allocate(x(1:n))
	  x = m
	end subroutine

	program ppp
	interface
	 subroutine  sub1(x,n,m)
	  integer x(:),n,m
	  pointer x
	 end subroutine
	 subroutine  sub2(x,n,m)
	  pointer x
	  integer x(:),n,m
	 end subroutine
	 subroutine  sub3(x,n,m)
	  integer,pointer:: x(:)
	  integer n,m
	 end subroutine
	 subroutine  sub4(x,n,m)
	  integer,dimension(:),pointer:: x
	 end subroutine
	 subroutine  sub5(x,n,m)
	  integer,pointer,dimension(:):: x
	  integer n,m
	 end subroutine
	end interface

	integer, pointer, dimension(:) :: x1,x2,x3,x4,x5
	integer, dimension(5) :: result,expect
	data expect/10,10,9,40,10/

	call sub1(x1,10,1)
	call sub2(x2,5,2)
	call sub3(x3,3,3)
	call sub4(x4,20,2)
	call sub5(x5,2,5)

	result(1) = sum(x1)
	result(2) = sum(x2)
	result(3) = sum(x3)
	result(4) = sum(x4)
	result(5) = sum(x5)

	call check(result,expect,5)
	end program
