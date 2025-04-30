! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! check for constant expression : 1.0q0.

program test
  real*16 :: r
  !{error "PGF90-S-0084-Illegal use of symbol  - KIND parameter"}
  r = 1.23456789123456789123456789123456789q0_16

end program
