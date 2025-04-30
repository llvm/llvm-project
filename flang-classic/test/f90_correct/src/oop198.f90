! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       

program unlimited_poly
USE CHECK_MOD
  type my_type
  integer i 
  end type

  type(my_type) z
  logical,target :: l 
  logical results(4)
  logical expect(4)

  class(*),allocatable :: a
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
  allocate(a,source=z%i)
  select type(a)
  type is(integer)
  results(4) = a .eq. z%i
  end select
 
  call check(results,expect,4)
  
end program unlimited_poly


