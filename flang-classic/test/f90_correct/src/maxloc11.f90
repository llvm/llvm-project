!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! maxloc with back in parameter declaration

program main
  integer, parameter :: a(1) = maxloc((/1,4,4,1/),back=.true.)
  call check(a(1), 3, 1)
end
