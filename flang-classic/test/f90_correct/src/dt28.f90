!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!   type constructors with pointer members including a recursive derived
!   where the constructor contains derived type refs
!
	module m
	 type dt
	  integer :: size
	  integer,dimension(:),pointer :: value
	  type(dt),pointer:: left,right
	 end type
	end module

	program p
	 use m
	 type(dt):: a,b
	 type(dt),pointer :: head
	 integer, dimension(6) :: r1, e1
	 integer, dimension(10) :: r2a, r2b, e2
	 integer, dimension(5) :: r3a, r3b, e3
	 data e1 / 10,5,0,10,5,0 /
	 data e2 / 2,4,6,8,10,12,14,16,18,20 /
	 data e3 / 10,20,30,40,50 /

	 a%size = 10
	 allocate(a%value(a%size))
	 do i = 1,a%size
	  a%value(i) = 2*i
	 enddo

	 allocate(a%left)
	 a%left%size = 5
	 allocate(a%left%value(a%left%size))
	 do i = 1,a%left%size
	  a%left%value(i) = 10*i
	 enddo
	 nullify(a%right)

	 b = dt( a%size, a%value, a%left, a%right )

	 r1(1) = b%size
	 r2a = b%value
	 if( associated(b%left) ) then
	  r1(2) = b%left%size
	  r3a = b%left%value
	 else
	  r1(2) = -1
	 endif
	 if( associated(b%right) ) then
	  r1(3) = -1
	 else
	  r1(3) = 0
	 endif

	 allocate(head)

	 head = dt( a%size, a%value, a%left, a%right )
	 r1(4) = head%size
	 r2b = head%value
	 if( associated(head%left) ) then
	  r1(5) = head%left%size
	  r3b = head%left%value
	 else
	  r1(5) = -1
	 endif
	 if( associated(head%right) ) then
	  r1(6) = -1
	 else
	  r1(6) = 0
	 endif

	call check( r1, e1, 6 )
	call check( r2a, e2, 10 )
	call check( r2b, e2, 10 )
	call check( r3a, e3, 5 )
	call check( r3b, e3, 5 )
	
	end program
