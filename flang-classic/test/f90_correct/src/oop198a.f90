! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       
! Same as oop198.f90 except it uses multiple allocatable targets.

program unlimited_poly
!USE CHECK_MOD
  type my_type
  integer i 
  end type

  type(my_type) z
  logical,target :: l 
  logical results(5)
  logical expect(5)

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
  allocate(a,a2,source=z%i)
  select type(a2)
  type is (integer)
  select type(a)
  type is(integer)
  results(4) = a .eq. z%i
  results(5) = a .eq. a2
  end select
  end select
 
  call check(results,expect,5)
  
end program unlimited_poly


