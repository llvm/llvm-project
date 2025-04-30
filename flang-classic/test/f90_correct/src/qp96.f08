! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test WRITE of implicit real variable initialized with quad-precision value

program main
  character(len=32) :: str1
  x = 1.0_16 + 2.0_16**(-105)
  write (str1,'(z0)') 'X'

  write(*,*) 'PASS'

end program main
