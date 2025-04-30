! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       

logical function check_logical(p) result (RSLT)
  class(*),pointer :: p
  select type(p)
  type is (logical)
     RSLT = .true.
     class default
     RSLT = .false.
  end select
end function check_logical

logical function check_alloc(a) result (RSLT)
  class(*),allocatable :: a
  select type(a)
  type is (integer*8)
     RSLT = .true.
     class default
     RSLT = .false.
   end select
end function check_alloc


program unlimited_poly
USE CHECK_MOD
  
  logical,target :: l 
  logical results(9)
  logical expect(9)
  integer*8,target:: z
  class(*),pointer :: lp
  
  interface
     logical function check_logical(p) result (RSLT)
       class(*),pointer :: p
     end function check_logical
     
     logical function check_alloc(a) result (RSLT)
       class(*),allocatable :: a
     end function check_alloc
  end interface
  
  type my_type
     class(*),allocatable :: a
     class(*),pointer :: p
  end type my_type
  
  type(my_type) :: obj

  results = .false.
  expect = .true.
  
  
  l = .false.

  allocate(integer*8::obj%a)

  obj%p => l

  select type(p=>obj%p)
  type is (logical)
     results(1) = .true.
     p = .true.
  end select 

  select type(a=>obj%a)
  type is (integer*8)
     results(2) = .true.
  type is (integer)
     results(2) = .false.
  end select

  results(3) = l

  lp => z
  results(8) = same_type_as(obj%a,lp)
  results(9) = extends_type_of(obj%a,lp)

  lp => l
  results(6) = same_type_as(obj%p,lp)
  results(7) = extends_type_of(obj%p,lp)
  
  results(4) = check_logical(obj%p)
  results(5) = check_alloc(obj%a)

  
  call check(results,expect,9)
  
end program unlimited_poly


