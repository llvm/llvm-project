! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test ptr reshape (F2003) with derived types

	program test
	implicit none
	parameter (n = 3)
	integer n
	type nested_pointer
	real, pointer :: diag2(:)
	end type nested_pointer
	type pointers
	real, pointer :: base_array(:), diagonal(:)
	real, pointer :: matrix(:,:)
	type(nested_pointer) :: np
	character, pointer :: cp(:)
	character,pointer :: cp2(:)
	character(len=3), pointer :: cp4(:)
	end type pointers
	type(pointers) :: ptrs	
	character(len=3), pointer :: cp3(:)
	character(len=24) :: exp_test12
        character(len=24) :: rslt_test12
	integer expect(12)
	integer result(12)
	integer test1(n,n)
        integer test2(n)
	integer test3(n)
	integer test4(n)
	integer test5(n)
	integer shapeArr(2)
	character, target :: test10(8)
	character(len=3), target :: test12(8)
	integer i,j
	data test10 /'z','y','x','v','u','t','s','r'/
	data test12 /'abc','def','ghi','jkl','mno','pqr','stu','vwx'/

	exp_test12 = 'abcdefghijklmnopqrstuvwx'
	rslt_test12 = ''
	allocate(ptrs%base_array(n*n))
	expect = .true.
	result = .false.
	shapeArr(1) = n
	shapeArr(2) = n
	do i=1,n*n
	  ptrs%base_array(i) = i-1
	enddo
	test1 = reshape(ptrs%base_array,shapeArr)
	ptrs%matrix(1:n,1:n) => ptrs%base_array
	result(1) = all(ptrs%matrix .eq. test1)
	ptrs%diagonal(1:n) => ptrs%base_array(::n+1)
	test2 = ptrs%base_array(::n+1)
	result(2) = all(ptrs%diagonal .eq. test2)
	ptrs%diagonal(n-1:) => ptrs%base_array(::n+1)
	test3 = ptrs%base_array(::n+1)
	result(3) = all(ptrs%diagonal .eq. test3)
	ptrs%diagonal(1:n) => ptrs%base_array(n*n:1:-(n+1))
	result(4) = all(ptrs%diagonal .eq. test2(n:1:-1)) 	
	ptrs%np%diag2(n:1) => ptrs%diagonal
	result(5) = all(ptrs%np%diag2 .eq. test2)
	ptrs%diagonal(n:1) => ptrs%diagonal
	result(6) = all(ptrs%diagonal .eq. test2)
	ptrs%diagonal(n:1) => ptrs%base_array(n*n:1:-(n+1))
	result(7) = all(ptrs%diagonal .eq. test2)
	ptrs%diagonal(1:n) => ptrs%matrix(:,n)
	test4 = ptrs%matrix(:,n)
	result(8) = all(ptrs%diagonal .eq. test4)
	ptrs%diagonal(1:) => ptrs%matrix(n,:)
	test5 = ptrs%matrix(n,:)
	result(9) = all(ptrs%diagonal .eq. test5)
	ptrs%cp2 => test10	
	ptrs%cp(8:1) => ptrs%cp2
	result(10) = all(ptrs%cp .eq. test10(8:1:-1))
	ptrs%cp(4:8) => ptrs%cp
	result(11) = all(ptrs%cp .eq. test10(8:4:-1))
	ptrs%cp4(8:1) => test12(8:1:-1)
	cp3(1:8) => ptrs%cp4	
	rslt_test12 = cp3(1)//cp3(2)//cp3(3)//cp3(4)//cp3(5)//cp3(6)//cp3(7)//cp3(8)
	result(12) = rslt_test12 .eq. exp_test12
	
	call check(result,expect,12)
	end
	
	

		
