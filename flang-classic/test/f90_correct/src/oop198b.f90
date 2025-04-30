! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       
! Same as oop198a.f90 except it uses multiple mold= allocations

program unlimited_poly
!USE CHECK_MOD
  type my_type
  integer i 
  end type

  type(my_type) z
  logical,target :: l 
  logical results(3)
  logical expect(3)

  class(*),allocatable :: a,a2
  class(*),pointer :: p
  logical, pointer :: lp

  results = .false.
  expect = .true.
  

  l = .false.

  allocate(integer*8::a)

  p => l

  select type(p)
  type is (logical)
  results(1) = .true.
  p = .true.
  end select 

  select type(a)
  type is (integer*8)
  results(2) = .true.
  type is (integer)
  results(2) = .false.
  end select

  results(3) = l

  z%i = 777
  deallocate(a)
  allocate(a,a2,mold=z%i)
  select type(a2)
  type is (integer)
  end select
 
  call check(results,expect,3)
  
end program unlimited_poly


