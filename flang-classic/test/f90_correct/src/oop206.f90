! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       

program unlimited_poly
USE CHECK_MOD

  type my_type
  class(*),allocatable :: a
  class(*),pointer :: p
  end type

  integer z 
  logical,target :: l 
  logical results(2)
  logical expect(2)
  type(my_type) :: obj
  type(my_type),allocatable :: obj2

  class(*), allocatable :: u2(:)
  integer sa(5)
  integer, allocatable :: sa2(:)
  integer i,j,k

  do i=1,5
      sa(i) = i
 enddo

  results = .false.
  expect = .true.
  


   allocate(u2(5),source=sa)
   allocate(sa2(5))
   sa2 = -1

   select type(qq=>u2)
   type is (integer)
   print *, qq,sa2
   sa2 = qq
   results(1) = all(sa2 .eq. qq)
   end select
   print *, sa2

   results(2) = all(sa2 .eq. sa)

  call check(results,expect,2)
  
end program unlimited_poly


