! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       

program unlimited_poly
USE CHECK_MOD
  
  logical,target :: l 
  logical results(5)
  logical expect(5)
  integer*8,target:: z
  class(*),pointer :: lp
  
  type my_type
     class(*),allocatable :: a
     class(*),pointer :: p
  end type my_type
  
  type(my_type) :: obj

  results = .false.
  expect = .true.
   
  l = .true.
  lp => l 
  obj%p => l

  results(1) = same_type_as(lp,obj%p)

  select type (p=>lp)
  type is (logical)
  !print *, lp
  results(4) = p
  end select

  select type (p=>obj%p)
  type is (logical)
  !print *, p
  results(2) = p
  end select

  select type (lp)
  type is (logical)
  !print *, lp
  results(3) = lp
  end select

  call check(results,expect,4)
  
end program unlimited_poly


