! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! host routine count intrinsic with contained routine count variable

integer n
n = count([.true., .false., .true.])
call ss(n)

contains
  subroutine ss(n)
    integer :: count
    count = n
    if (count .ne. 2) print*, 'FAIL'
    if (count .eq. 2) print*, 'PASS'
  end
end
