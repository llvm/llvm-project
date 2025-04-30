! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

program test 

   use, intrinsic :: iso_c_binding 

   implicit none 

   real, pointer :: ptr(:,:) => NULL() 
   real, pointer :: uin(:,:,:) => NULL() 

   integer, parameter :: nx = 4 
   integer, parameter :: ny = 5 

   integer :: i, j 

   allocate(ptr(nx,ny)) 

   forall (i=1:nx,j=1:ny) ptr(i,j) = 10*i+j 

   write (*,*) 'shape(ptr): ', shape(ptr) 
   write (*,'(a,x,z32)') ' loc(ptr(1,1)): ', loc(ptr(1,1)) 
   write (*,*) 'ptr: ', ptr 

   call c_f_pointer(c_loc(ptr(1,1)),uin,[nx,ny,1]) 

   write (*,*) 'shape(uin): ', shape(uin) 
   write (*,'(a,x,z32)') ' loc(uin(1,1,1)): ', loc(uin(1,1,1)) 
   write (*,*) 'uin: ', uin 

   deallocate(ptr) 

   print *, "PASS"

end program test 

