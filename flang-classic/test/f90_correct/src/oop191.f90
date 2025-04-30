! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       

program unlimited_poly
USE CHECK_MOD
  
  logical,target :: l 
  logical results(3)
  logical expect(3)

  type my_type
  class(*),allocatable :: a
  class(*),pointer :: p
  end type

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
 
  call check(results,expect,3)
  
end program unlimited_poly


