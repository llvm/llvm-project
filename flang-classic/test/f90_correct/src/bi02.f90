!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!


	module test

        integer N
	parameter (  N = 77)

	integer , BIND(c) :: c
	common /result/ r_array
	integer r_array(N)
	BIND (c) ::/result/

	common /expect/ e_array
	integer e_array(N)
	BIND (c) ::/expect/

	end

	program main
	
	use iso_c_binding
	use test



	interface
	

    	subroutine check (a,b,c) bind (c)
        integer a(N), b(N), c
        end subroutine check


	subroutine c_call_other (cc) BIND(C, name = 'c_call')
	use iso_c_binding
	type(C_PTR), VALUE :: cc
	end subroutine c_call_other

	subroutine c_call_ref (cc) BIND(C)
	use iso_c_binding
	type(C_PTR) :: cc
	end subroutine c_call_ref

	function get_int (cc) BIND(c)
	use iso_c_binding
        type(C_PTR), VALUE :: cc
        integer  get_int
        end function get_int 

	function get_ptr (cc) BIND(c)
	use iso_c_binding
        type(C_PTR), VALUE :: cc
        type(C_PTR)  get_ptr
        end function get_ptr 

	function get_ptr_ref_foo (cc) BIND(c, name = "get_ptr_ref")
	use iso_c_binding
        type(C_PTR) :: cc
        type(C_PTR)  get_ptr_ref_foo
        end function get_ptr_ref_foo


	subroutine cp_call (ii, cp, kk, i2) BIND(C)
	use iso_c_binding
	INTEGER , VALUE :: ii
	type(C_PTR), VALUE :: cp
	type(C_PTR), VALUE :: kk 
	INTEGER , VALUE :: i2
	end subroutine cp_call


	type(C_PTR)  function c_fun (c2) BIND (c)
	use iso_c_binding
	type(C_PTR), value :: c2
	end function c_fun

	type(C_PTR)  function c_fun_ref (c2) BIND (c)
	use iso_c_binding
	type(C_PTR) :: c2
	end function c_fun_ref

	TYPE(C_PTR) function c_fun_ptr(c2) BIND (c)
	use iso_c_binding
	TYPE(C_PTR) , value :: c2
	end function c_fun_ptr

	type(C_PTR) function cp_fun (ii,cp2, cp3, i2) BIND (c)
	use iso_c_binding
	INTEGER,  value :: ii 
	type(C_PTR), value :: cp2
	type(C_PTR), value :: cp3
	INTEGER,  value :: i2 
	end function cp_fun
	end interface


	type (C_PTR),TARGET :: p1
	type (C_PTR) :: p2
	type (C_PTR) :: p3
	type (C_PTR) :: pret

	integer, TARGET :: F1
	integer, TARGET :: F2
	integer ii
        integer, target, dimension (10,10):: ARR

	type my_type 
		real    r
		integer i
		type (C_PTR) pptr
		type (C_PTR),dimension (10,10) :: amtr
	end type my_type

	type (C_PTR),dimension (10,10) :: aptr
	type (my_type),dimension (10,10) :: struct_aptr


	type (my_type) mt

	F1 = 12
	f2 = 3
        ii = 5
        ARR(5,5) = 55
        ARR(1,1) = 11


	p2 = C_LOC(F2)
	p1 = C_LOC(F2)
        mt%pptr = C_LOC(F1)
        aptr(2,2) = C_LOC(F1)
        struct_aptr(2,2)%pptr = C_LOC(F1)
        struct_aptr(2,2)%amtr(3,3) = C_LOC(F1)
	mt%i = 33

	pret = c_fun (C_LOC(F2))
	pret = c_fun (C_LOC(ARR(5,5)))
	pret = c_fun (C_LOC(ARR))
	pret = c_fun (C_LOC(mt%i))
	pret = c_fun (p1)
	pret = c_fun (mt%pptr)
	pret = c_fun (aptr(2,2))
	pret = c_fun (struct_aptr(2,2)%pptr)
	pret = c_fun (struct_aptr(2,2)%amtr(3,3))
	pret = c_fun(get_ptr(p1))

	call c_call_other(pret)
	call c_call_other(get_ptr(C_LOC(F2)))
	p2 = get_ptr(C_LOC(F2));
	p2 = get_ptr(p1);
	call c_call_other (p2);

!	print * , " now by reference "

!       Find out what test number here:
!	print * , c
	call c_call_ref(get_ptr(C_LOC(F2)))
	p2 = get_ptr_ref_foo(p1);
	p2 = get_ptr(C_LOC(F2));
	p2 = get_ptr_ref_foo(C_LOC(F2))
	call c_call_other (p2)
!       Find out what test number here:
!	print * , c
	p2 = C_LOC(F1)
	call c_call_other(get_ptr(p2))

	p2 =  get_ptr(p2)
	call c_call_other (p2)
	p2 =  get_ptr(get_ptr(C_LOC(F2)))
	p2 =  get_ptr_ref_foo(get_ptr_ref_foo(C_LOC(F2)))
	p2 =  get_ptr(get_ptr_ref_foo(C_LOC(F2)))
	p2 =  get_ptr_ref_foo(get_ptr(C_LOC(F2)))

	p2 =  get_ptr(get_ptr(p1))
	p2 =  get_ptr_ref_foo(get_ptr_ref_foo(p1))
	p2 =  get_ptr(get_ptr_ref_foo(p1))
	p2 =  get_ptr_ref_foo(get_ptr(p1))

	call c_call_other(get_ptr(p1))
	call c_call_other(get_ptr_ref_foo(p1))
	call c_call_ref(get_ptr(p1))
	call c_call_ref(get_ptr_ref_foo(p1))

	call c_call_other ( C_LOC(F1))
	call c_call_other (C_LOC(ARR(5,5)))
	call c_call_other (C_LOC(ARR))
	call cp_call ( ii, p1, C_LOC(F1), ii)
	p2 = cp_fun ( 5, p1, C_LOC(F1),ii)
	call c_call_other (C_LOC(mt%i))
	call c_call_other (p1)
	call c_call_other (mt%pptr)
	call c_call_other (aptr(2,2))
 	call c_call_other (struct_aptr(2,2)%pptr)
	call c_call_other (struct_aptr(2,2)%amtr(3,3))
	call c_call_other (get_ptr(p1))



	p2 = cp_fun (ii,p1,C_LOC(F1),5)


	p2 = c_fun_ptr (C_LOC(p1))
	p2 = C_LOC(p1)
	p1 = c_fun_ptr(p2)
!	 simple struct assignment still works 
	p1 = C_LOC(f2)
	p2 = p1

	p3 = c_fun(p2)
	call check (r_array, e_array, N)
	end
	
       
