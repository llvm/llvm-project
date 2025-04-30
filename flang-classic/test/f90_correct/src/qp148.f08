! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test I/O for quad complex

program test
  integer, parameter :: k = 16
  real(kind = k) :: r(3)
  complex(kind = k) :: c(3)

  r(3) = 1.41242153123125123512351243143214_16
  c(3) = (0.1212151353124143234324134134143_16, -5.12312512531361532514324153135155221_16)

  write(*, *) r(3), c(3)
  write(*, *) 'PASS'
end program
