! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test exponentiation of quad-precision values

program main
   integer, parameter :: n = 7 
   real(16) :: q_tol = 5e-33_16
   real(16) :: res(n), expct(n)
   expct(1) = 2.00000000000000000000000000000000000_16 
   expct(2) = 2.00000000000000000000000000000000000_16
   expct(3) = 1.18920711500272106671749997056047593_16
   expct(4) = 1.41421356237309504880168872420969798_16
   expct(5) = 1.68179283050742908606225095246642969_16
   expct(6) = -5.65685424949238019520675489683879194_16
   expct(7) = -2.82842712474619009760337744841939597_16

   res(1) = 2.0_16 ** 1_4 
   res(2) = 2.0_16 ** 1_8 
   res(3) = 2.0_16 ** 0.25_16
   res(4) = 2.0_16 ** 0.5_16
   res(5) = 2.0_16 ** 0.75_16
   res(6) = -2.0_16 ** 2.5_16
   res(7) = -2.0_16 ** 1.5_16
 
  do i = 1, n
    if(expct(i) .eq. 0.0_16) then
      if (res(i) .ne. expct(i)) STOP i  
    else
      if (abs((res(i) - expct(i)) / expct(i)) .gt. q_tol) STOP i
    endif
  enddo
  print *, 'PASS'
end
