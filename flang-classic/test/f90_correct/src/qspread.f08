! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.                    
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                      
!                                                                              
! test SPREAD intrinsic with quad-precision arguments

program main
   implicit none
   integer, parameter :: q = 16
   real(kind = q), dimension(10) :: i_q
   real(kind = q), dimension (2, 3) :: a_q
   real(kind = q), dimension (2, 2, 3) :: b_q
   character (len=800) res1, res2, res3

   a_q = reshape ((/1.0_q, 2.0_q, 3.0_q, 4.0_q, 5.0_q, 6.0_q/), (/2, 3/))
   b_q = spread (a_q, 1, 2)
   if (any (b_q .ne. reshape ((/1.0_q, 1.0_q, 2.0_q, 2.0_q, 3.0_q, 3.0_q, 4.0_q, 4.0_q, 5.0_q, 5.0_q, 6.0_q, 6.0_q/), &
                            (/2, 2, 3/)))) &
      STOP 1
   res1 = ' '
   write(res1, 100) b_q
   res2 = ' '
   write(res2, 100) spread (a_q, 1, 2)
   if (res1 /= res2) STOP 2
   res3 = ' '
   write(res3, 100) spread (a_q, 1, 2) + 0.0_q
   if (res1 /= res3) STOP 3
   i_q = spread(1.0_q, 1, 10)
   if (any(i_q /= 1.0_q)) STOP 4

   100 format(12F35.34)
   write(*,*) 'PASS'
end program main
