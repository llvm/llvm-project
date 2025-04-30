! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       
! Same as oop205 except it uses multiple sourced allocations.

program unlimited_poly
!USE CHECK_MOD

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
  


   allocate(u2(5,5),sa2,source=sa)
   !allocate(sa2(5,5))
   !sa2 = sa

   select type(qq=>u2)
   type is (integer)
   print *, sa2
   sa2 = qq
   i = size(qq)
   results(1) = i .eq. 25
   results(2) = all(sa2 .eq. qq)
   results(3) = all(qq .eq. sa)
   end select
   results(4) = all(sa2 .eq. sa)

  call check(results,expect,4)
  
end program unlimited_poly


