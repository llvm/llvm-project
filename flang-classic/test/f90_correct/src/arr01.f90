! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for use deferrd shape array which declared wrongly

program test
  !{error "PGF90-S-0084-Illegal use of symbol a - a named constant array must have constant extents"}
  integer, parameter :: a(:) = 1
  integer, parameter :: b(4) = reshape([a, a+1, a+2, a+3], shape(b))
end program test
