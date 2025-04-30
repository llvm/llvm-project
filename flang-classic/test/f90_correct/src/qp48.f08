! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! check that I/O of highest-precision (quad) real numbers do not introduce errors larger than epsilon

program test
  use iso_fortran_env, only: real_kinds
  implicit none
  integer, parameter :: qcon = real_kinds(ubound(real_kinds,dim=1))
  real(kind = qcon) :: a, b(2), c
  integer :: exponent
  character(len = 180) :: src

  exponent = 4000
  b(:) = huge (1.0_qcon)/10.0_qcon**exponent
  write (src, *) b
  read (src, *) a, c
  if (abs (a-b(1))/a > epsilon(0.0_qcon) &
      .or. abs (c-b(1))/c > epsilon (0.0_qcon)) STOP 1
  write(*,*) 'PASS'

end program test

