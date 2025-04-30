!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! minloc with back in parameter declaration

program main
  integer, parameter :: a(1) = minloc((/1,4,4,1/),back=.true.)
  call check(a(1), 4, 1)
end
