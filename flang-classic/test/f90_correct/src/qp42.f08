! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test WRITE statement with quad-precision input

program test
  use check_mod
  integer, parameter :: k = 16
  real(kind = k) :: tmpa(4)
  character(200) :: str, estr
  integer :: rslt, expct

  data tmpa / -1.123456789123456789123456789123456789_16,    &
              -0.000123456789123456789123456789123456789_16, &
               1.123456789123456789123456789123456789_16,    &
               0.000123456789123456789123456789123456789_16  /


  estr = "  -1.12345678912345678912345678912345676        -1.23456789123456789123456789123456789E-0004   1.12345678912345678912345678912345676         1.23456789123456789123456789123456789E-0004"
  rslt = 0
  expct = -1

  write(str,*) tmpa
  rslt = str == estr

  call checki4(rslt, expct, 1)

end program
