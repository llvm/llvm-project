! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test public NAMELISTs with private quad-precision values

program main
  type :: tp2
    real(kind=16) :: t
  end type

  type :: tp1
    real(kind=16) :: t
  end type

  type(tp1) t1
  type(tp2) t2
  namelist /myname/t1,t2
  t1%t=1.123456789123456789123456789_16
  t2%t=1.123456789123456789123456789_16

  write(*,NML=myname)
  write(*,*) 'PASS'

end program
