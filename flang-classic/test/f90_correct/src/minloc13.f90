!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! minloc in initialization with repeat real elements

program p
integer :: rslt = 2
integer, parameter, dimension(1) :: mn = minloc((/4.5, 3.1, 6.8, 3.1/))
call check(mn, rslt, 1)
end
