! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! undeclared intrinsic reference in an internal routine

character*4 :: s = 'FAIL'
if (f1() + f2() .eq. 5) s = 'PASS'
print*, s

contains
  integer function f1
    integer :: count
    f1 = count([.true., .false., .true.])
  end

  integer function f2
    count = 3
    f2 = count
  end
end
