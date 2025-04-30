!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! maxloc in initialization with repeat real elements

program p
integer :: rslt = 2
integer, parameter, dimension(1) :: mn = maxloc((/4.5, 6.8, 6.8, 3.1/))
call check(mn, rslt, 1)
end
