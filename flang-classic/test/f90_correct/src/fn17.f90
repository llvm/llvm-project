!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Problem when a function call is used as a loop upper bound,
! and the loop contains code that will create an allocatable
! temp, and the allocate of the temp gets floated out of the loop
! the function call is replicated three times: once as the loop limit,
! once as the zero-trip test for the allocate, and again for the deallocate
! problem manifested itself as far back as 6.1 pgf90, maybe earlier

module mmm1
 type dt
  integer :: pos(3)
 end type

 integer :: ii = 0
contains

 integer function n1(self)
  type(dt),dimension(:) :: self
  ii = ii + 1
  n1 = size(self) - ii
 end function

 subroutine rotate1(self,matrix)
  type(dt),dimension(:) :: self
  integer,intent(in),dimension(3,3)::matrix
  integer:: n
  do n = 1,n1(self)
   self(n)%pos = matmul(matrix,self(n)%pos)
  enddo
 end subroutine

end module

program p
 use mmm1
 type(dt),dimension(10) :: self
 integer,dimension(3,3) :: matrix
 integer::result(30)
 integer::expect(30)
 data expect/ 1,3,2,1,3,2,1,3,2,1, &
              2,1,3,2,1,3,2,1,3,2, &
              3,2,1,3,2,1,3,2,1,3 /

 do i = 1,10
  self(i)%pos(1) = 1
  self(i)%pos(2) = 2
  self(i)%pos(3) = 3
 enddo

 matrix(1,:) = (/ 0, 1, 0 /)
 matrix(2,:) = (/ 0, 0, 1 /)
 matrix(3,:) = (/ 1, 0, 0 /)

 do i = 1,10
  call rotate1( self, matrix )
 enddo
 result( 1:10) = self(:)%pos(1)
 result(11:20) = self(:)%pos(2)
 result(21:30) = self(:)%pos(3)
 call check(result,expect,30)
end program
