! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test ptr reshape (F2003)

	program test
	implicit none
	parameter (n = 3)
	integer n
	real, pointer :: base_array(:), matrix(:,:), diagonal(:)
	real, pointer :: diag2(:)
	character, pointer :: cp(:)
	character,pointer :: cp2(:)
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
	allocate(base_array(n*n))
	expect = .true.
	result = .false.
	shapeArr(1) = n
	shapeArr(2) = n
	do i=1,n*n
	  base_array(i) = i-1
	enddo
	test1 = reshape(base_array,shapeArr)
	matrix(1:n,1:n) => base_array
	result(1) = all(matrix .eq. test1)
	diagonal(1:n) => base_array(::n+1)
	test2 = base_array(::n+1)
	result(2) = all(diagonal .eq. test2)
	diagonal(n+1:) => base_array(::n+1)
	test3 = base_array(::n+1)
	result(3) = all(test3 .eq. diagonal)
	diagonal(1:n) => base_array(n*n:1:-(n+1))
	result(4) = all(diagonal .eq. test2(n:1:-1)) 	
	diag2(n:1) => diagonal
	result(5) = all(diag2 .eq. test2)
	diagonal(n:1) => diagonal
	result(6) = all(diagonal .eq. test2)
	diagonal(n:1) => base_array(n*n:1:-(n+1))
	result(7) = all(diagonal .eq. test2)
	diagonal(2:n-1) => matrix(:,n)
	test4 = matrix(:,n)
	result(8) = all(diagonal .eq. test4)
	diagonal(-1:) => matrix(n,:)
	test5 = matrix(n,:)
	result(9) = all(diagonal .eq. test5)
	cp2 => test10	
	cp(8:1) => cp2
	result(10) = all(cp .eq. test10(8:1:-1))
	cp(4:8) => cp
	result(11) = all(cp(::-1) .eq. test10(5:1:-1))
	cp3(8:1) => test12(8:1:-1)
	rslt_test12 = cp3(1)//cp3(2)//cp3(3)//cp3(4)//cp3(5)//cp3(6)//cp3(7)//cp3(8)
	result(12) = rslt_test12 .eq. exp_test12
	call check(result,expect,12)
	end
	
	

		
