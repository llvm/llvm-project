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
  logical results(4)
  logical expect(4)
  type(my_type) :: obj
  type(my_type),allocatable :: obj2

  class(*), allocatable :: u2(:,:)
  class(*), allocatable :: u3(:,:)

  integer sa(5,5)
  integer, allocatable :: sa2(:,:)
  integer i,j,k

  k = 0
  do i=1,5
   do j=1, 5
      sa(i,j) = k
      k = k + 1
   enddo
 enddo

  results = .false.
  expect = .true.
  
  allocate(sa2(5,5))
   sa2 = sa

  ALLOCATE (u3(lbound(sa2,1):ubound(sa2,1),lbound(sa2,2):ubound(sa2,2)),SOURCE=sa2)
  allocate(u2(lbound(sa2,1):ubound(sa2,1),lbound(sa2,2):ubound(sa2,2)),source=u3)


   select type(qq=>u2)
   type is (integer)
   print *, sa2
   print *, qq
   sa2 = qq
   i = size(qq)
   print *, i
   results(1) = i .eq. 25
   results(2) = all(sa2 .eq. qq)
   results(3) = all(qq .eq. sa)
   end select
   results(4) = all(sa2 .eq. sa)


  call check(results,expect,4)
  
end program unlimited_poly


