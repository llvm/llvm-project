! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

program test 

   use  iso_c_binding 

   implicit none 

   integer, pointer :: ptr(:,:) => NULL() 
   integer, pointer :: uin(:,:) => NULL() 
   integer, pointer::pp

   integer, parameter :: nx = 4 
   integer, parameter :: ny = 5 
   type(c_ptr)::foo

   integer :: i, j 

   allocate(ptr(nx,ny)) 

   forall (i=1:nx,j=1:ny) ptr(i,j) = 10*i+j 

    pp=>ptr(1,1)
   call c_f_pointer(c_loc(pp),uin,[nx,ny]) 

   write (*,*) 'shape(uin): ', shape(uin) 
   write (*,'(a,x,z32)') ' loc(uin(1,1)): ', loc(uin(1,1)) 
   write (*,*) 'uin: ', uin 



   deallocate(ptr) 
   print *, "PASS"

end program test 

subroutine dummy(i)
integer i
print *, i

end

