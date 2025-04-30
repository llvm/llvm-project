!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test f2008 is_contiguous inquiry intrinsic

program p 
   integer, parameter :: N = 8
   logical :: expect(n) = (/.FALSE.,.TRUE.,.FALSE.,.FALSE.,.TRUE.,.FALSE.,.FALSE.,.TRUE./)
   logical :: result(n)
   real, target :: a(10,10)
   real, pointer :: a_ptr(:,:)
   integer :: i
   result(1) = is_contiguous(a_ptr)

   a_ptr=>a
   result(2) = is_contiguous(a_ptr)

   NULLIFY(a_ptr)
   result(3) = is_contiguous(a_ptr)

   a_ptr=>a(1:10:2,1:20:2)
   result(4) = is_contiguous(a_ptr)

   a_ptr=>a(1:10,1:5)
   result(5) = is_contiguous(a_ptr)

   a_ptr=>a(2:10,1:5)
   result(6) = is_contiguous(a_ptr)

   a_ptr=>a(2:10,2:10)
   result(7) = is_contiguous(a_ptr)

   allocate(a_ptr(1:5, 1:25))
   result(8) = is_contiguous(a_ptr)

   call check(result, expect, N)
   !print *, result
end program
