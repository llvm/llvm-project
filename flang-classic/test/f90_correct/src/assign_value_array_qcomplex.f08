! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! this test case is test for assigning values for quadruple precision complex array

program test
  complex(16), dimension(3,3) :: a, b, c, d
  a = (999999999.8888_16, -4564675431354534654.454554_16)
  b = a
  c = (0.0_16,0.0_16)

  d = a - b
  if (all(d == c) .neqv. .true.) STOP 1
  write(*,*) 'PASS'
end program test
